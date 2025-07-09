import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')


class PolaLinearAttention(nn.Module):   
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.sr_ratio = sr_ratio
        self.alpha = alpha

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim,
                             kernel_size=kernel_size, groups=head_dim,
                             padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, num_heads, 1, head_dim)))
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.kernel_function = nn.ReLU()

        # Positional encoding placeholder (dynamically sized later)
        self.pos_encoding_cache = {}

    def forward(self, x):
        B, N, C = x.shape
        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)

        # Downsample for key/value
        if self.sr_ratio > 1:
            h = w = int(N ** 0.5)
            assert h * w == N, f"Expected square input, got N={N}"
            x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        
        k, v = kv[0], kv[1]
        n = k.shape[1]

        # === Positional encoding with dynamic support ===
        if (n, C) not in self.pos_encoding_cache:
            self.pos_encoding_cache[(n, C)] = nn.Parameter(torch.zeros(1, n, C).to(x.device))
        k = k + self.pos_encoding_cache[(n, C)]

        # === Kernelized similarity ===
        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * torch.sigmoid(self.power)

        q = q / scale
        k = k / scale

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)

        q_pos = self.kernel_function(q) ** power
        q_neg = self.kernel_function(-q) ** power
        k_pos = self.kernel_function(k) ** power
        k_neg = self.kernel_function(-k) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z

        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)

        # === DWC feature enhancement ===
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n),
                                          size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)

        # Auto derive spatial shape
        spatial_tokens = v.shape[2]
        H = W = int(spatial_tokens ** 0.5)
        assert H * W == spatial_tokens, f"Cannot reshape {spatial_tokens} to square"

        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

        x = x + v
        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# === Test ===
if __name__ == '__main__':
    from calflops import calculate_flops
    RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 1, 128, 40, 40
    inputs = torch.randn(B, H * W, C).to(device)

    model = PolaLinearAttention(dim=C, sr_ratio=2).to(device)
    outputs = model(inputs)
    print(GREEN + f"inputs.shape = {inputs.shape}, outputs.shape = {outputs.shape}" + RESET)

    print("\n[Flops Summary]")
    flops, macs, _ = calculate_flops(model=model,
                                     input_shape=(B, H * W, C),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)

    print(RESET)
