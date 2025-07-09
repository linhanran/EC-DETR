#!/bin/bash
source ~/.bashrc
source activate deim
CUDA_VISIBLE_DEVICES=0 python train.py -c /public/home/linhanran2023/DEIM-main/configs/EC-DETR/EC-DETR.yml --seed=0
