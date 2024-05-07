# GPU=3 setting='--dataset cifar10_wo_val --model resnet34 --noise 0.2 --noise_type val_split_asymm'
# CUDA_VISIBLE_DEVICES=$GPU python3 main_NL.py $setting
# CUDA_VISIBLE_DEVICES=$GPU python3 main_PL.py $setting --max_epochs 720
# CUDA_VISIBLE_DEVICES=$GPU python3 main_pseudo1.py $setting --lr 0.1 --max_epochs 480 --epoch_step 192 288
# CUDA_VISIBLE_DEVICES=$GPU python3 main_pseudo2.py $setting --lr 0.1 --max_epochs 480 --epoch_step 192 288

#!/bin/bash
# GPU='0,1' setting='--dataset cifar10_wo_val --model resnet34 --noise 0.2 --noise_type val_split_symm_exc'
GPU=0 setting='--dataset cifar10_wo_val --model resnet34 --noise 0.2 --noise_type val_split_symm_exc'
# CUDA_VISIBLE_DEVICES=$GPU python3 main_NL.py $setting
# CUDA_VISIBLE_DEVICES=$GPU python main_NLNL.py $setting
# CUDA_VISIBLE_DEVICES=$GPU python main_NLNL_original.py $setting

CUDA_VISIBLE_DEVICES=$GPU python3 main_NLNL_7.py $setting
# CUDA_VISIBLE_DEVICES=$GPU python main_NLNL_test.py $setting
# CUDA_VISIBLE_DEVICES=$GPU python main_NLNL_continue.py $setting
# CUDA_VISIBLE_DEVICES=$GPU python3 main_NLNL_test.py $setting
# CUDA_VISIBLE_DEVICES=$GPU python3 main_NL_o.py $setting
# CUDA_VISIBLE_DEVICES=$GPU python3 main_NL_Real_Dataset.py $setting
# CUDA_VISIBLE_DEVICES=$GPU python3 main_NL_Real_Dataset_old.py $setting
# CUDA_VISIBLE_DEVICES=$GPU python3 main_PL.py $setting --max_epochs 720
# CUDA_VISIBLE_DEVICES=$GPU python3 main_PL_Real_Dataset.py $setting --max_epochs 150
# CUDA_VISIBLE_DEVICES=$GPU python3 main_pseudo1.py $setting --lr 0.1 --max_epochs 480 --epoch_step 192 288
# CUDA_VISIBLE_DEVICES=$GPU python3 main_pseudo2.py $setting --lr 0.1 --max_epochs 480 --epoch_step 192 288
