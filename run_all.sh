#!/bin/bash

# REG-CycleGAN_syn_real
python /home/kreitnerl/mrs-gan/raytune.py --dataroot /home/kreitnerl/mrs-gan/datasets/new_horizon_syn_real_2 --name REG-CycleGAN_syn_real --model cycleGAN_W_REG --val_path /home/kreitnerl/mrs-gan/results/baselines/syn_i_ --gpu_ids 6,7 --TTUR --n_critic 5 --gan_mode wasserstein --save_epoch_freq 100 --quiet --roi 361,713 --n_layers_D 3 --lambda_feat 2.8 --lambda_B 2 --display_freq 2500 --cbamG

# REG-CycleGAN_syn_ucsf
python /home/kreitnerl/mrs-gan/raytune.py --dataroot /home/kreitnerl/mrs-gan/datasets/new_horizon_syn_ucsf --name REG-CycleGAN_syn_ucsf --model cycleGAN_W_REG --val_path /home/kreitnerl/mrs-gan/results/baselines/syn_i_ --gpu_ids 6,7 --TTUR --n_critic 5 --gan_mode wasserstein --save_epoch_freq 100 --quiet --roi 361,713 --n_layers_D 3 --lambda_feat 2.8 --lambda_B 2 --display_freq 2500 --cbamG

# REG-CycleGAN_ucsf
python /home/kreitnerl/mrs-gan/raytune.py --dataroot /home/kreitnerl/mrs-gan/datasets/new_horizon_ucsf --name REG-CycleGAN_ucsf --model cycleGAN_W_REG --val_path /home/kreitnerl/mrs-gan/results/baselines/syn_i_ --gpu_ids 6,7 --TTUR --n_critic 5 --gan_mode wasserstein --save_epoch_freq 100 --quiet --roi 361,713 --n_layers_D 3 --lambda_feat 2.8 --lambda_B 2 --display_freq 2500 --cbamG


# CycleGAN-WGP_ucsf
python /home/kreitnerl/mrs-gan/train.py --dataroot /home/kreitnerl/mrs-gan/datasets/new_horizon_ucsf --name CycleGAN-WGP_ucsf --model cycleGAN_W --val_path /home/kreitnerl/mrs-gan/results/baselines/syn_i_ --gpu_ids 6 --n_critic 5 --gan_mode wasserstein --save_epoch_freq 100 --quiet --roi 361,713 --lambda_identity 0.5 --cbamG &

# CycleGAN-WGP_syn_real
python /home/kreitnerl/mrs-gan/train.py --dataroot /home/kreitnerl/mrs-gan/datasets/new_horizon_syn_real --name CycleGAN-WGP_syn_real --model cycleGAN_W --val_path /home/kreitnerl/mrs-gan/results/baselines/syn_i_ --gpu_ids 6 --n_critic 5 --gan_mode wasserstein --save_epoch_freq 100 --quiet --roi 361,713 --lambda_identity 0.5 --cbamG &

# CycleGAN-WGP_syn_ucsf
python /home/kreitnerl/mrs-gan/train.py --dataroot /home/kreitnerl/mrs-gan/datasets/new_horizon_syn_ucsf --name CycleGAN-WGP_syn_ucsf --model cycleGAN_W --val_path /home/kreitnerl/mrs-gan/results/baselines/syn_i_ --gpu_ids 7 --n_critic 5 --gan_mode wasserstein --save_epoch_freq 100 --quiet --roi 361,713 --lambda_identity 0.5 --cbamG