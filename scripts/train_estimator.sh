#!/usr/bin/env bash


python train.py --dataroot '/home/john/SpectroscopyModel/dataset/preliminary_fitting_synthetic_dataset.mat' --file_ext '.mat' --name 'FittingModel_test_001' --phase 'train' --save_latest_freq 14000 --print_freq 14000 --no_flip --batchSize 50 --checkpoints_dir './tests' --niter 30 --niter_decay 0 --quiet --normalize --magnitude --actvn 'selu' --cropped_signal --lr 0.0001 --se --pAct --use_dropout --G0 --dataset_mode 'LabeledMatSpectralDataset'

python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext ['.mat','.h5'] --name 'FittingModel_test_001' --phase 'train' --save_latest_freq 14000 --print_freq 14000 --no_flip --batchSize 50 --checkpoints_dir './tests' --niter 30 --niter_decay 0 --quiet --normalize --magnitude --actvn 'selu' --cropped_signal --lr 0.0001 --se --pAct --use_dropout --G0 --dataset_mode 'LabeledMatSpectralDataset'

python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext '.mat' --name 'FittingModel_test_001' --phase 'train' --save_latest_freq 14000 --print_freq 14000 --no_flip --batchSize 50 --checkpoints_dir './tests' --niter 30 --niter_decay 0 --quiet --normalize --magnitude --actvn 'selu' --cropped_signal --lr 0.0001 --se --pAct --use_dropout --G0 --dataset_mode 'LabeledMatSpectralDataset'



python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext '.mat' --name 'FittingModel_test_016' --phase 'train' --save_latest_freq 28000 --print_freq 14000 --no_flip --batchSize 500 --checkpoints_dir './tests' --niter 50 --niter_decay 0 --quiet --normalize --magnitude --actvn 'relu' --cropped_signal --lr 0.0001 --se --pAct --G1 --dataset_mode 'LabeledMatSpectralDataset' --depth 2 --n_blocks 6 --use_dropout

# # Changed options
python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext '.mat' --name 'FittingModel_test_024' --phase 'train' --save_latest_freq 28000 --print_freq 14000 --no_flip --batchSize 500 --checkpoints_dir './tests' --niter 50 --niter_decay 0 --quiet --normalize --magnitude --actvn 'relu' --cropped_signal --lr 0.0001 --se --pAct --G1 --dataset_mode 'LabeledMatSpectralDataset' --depth 1 --n_blocks 6 --use_dropout --parameters

python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext '.mat' --name 'FittingModel_test_026' --phase 'train' --save_latest_freq 28000 --print_freq 14000 --no_flip --batchSize 500 --checkpoints_dir './tests' --niter 50 --niter_decay 0 --quiet --normalize --magnitude --actvn 'relu' --cropped_signal --lr 0.0001 --se --pAct --G2 --dataset_mode 'LabeledMatSpectralDataset' --depth 1 --n_blocks 6 --use_dropout --parameters

python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext '.mat' --name 'FittingModel_test_027' --phase 'train' --save_latest_freq 28000 --print_freq 14000 --no_flip --batchSize 500 --checkpoints_dir './tests' --niter 50 --niter_decay 0 --quiet --normalize --magnitude --actvn 'relu' --cropped_signal --lr 0.0001 --se --pAct --G2 --dataset_mode 'LabeledMatSpectralDataset' --depth 1 --n_blocks 6 --use_dropout --parameters --metabolites --labmda_metab 10








python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext '.mat' --name 'FittingModel_test_027' --phase 'train' --save_latest_freq 28000 --print_freq 14000 --no_flip --batchSize 500 --checkpoints_dir './tests' --niter 75 --niter_decay 0 --quiet --normalize --magnitude --actvn 'relu' --cropped_signal --lr 0.0001 --se --pAct --G2 --dataset_mode 'LabeledMatSpectralDataset' --depth 1 --n_blocks 6 --use_dropout --parameters --metabolites --lambda_metab 10 --phase_data_path '/home/john/SpectroscopyModel/tests/FittingModel_test_027/data'
python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext '.mat' --name 'FittingModel_test_028' --phase 'train' --save_latest_freq 28000 --print_freq 14000 --no_flip --batchSize 500 --checkpoints_dir './tests' --niter 75 --niter_decay 0 --quiet --normalize --magnitude --actvn 'relu' --cropped_signal --lr 0.0001 --se --pAct --G1 --dataset_mode 'LabeledMatSpectralDataset' --depth 1 --n_blocks 6 --use_dropout --parameters --metabolites --lambda_metab 100
python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext '.mat' --name 'FittingModel_test_029' --phase 'train' --save_latest_freq 28000 --print_freq 14000 --no_flip --batchSize 500 --checkpoints_dir './tests' --niter 75 --niter_decay 0 --quiet --normalize --magnitude --actvn 'relu' --cropped_signal --lr 0.0001 --se --pAct --G0 --dataset_mode 'LabeledMatSpectralDataset' --depth 1 --n_blocks 6 --use_dropout --parameters --metabolites --lambda_metab 1

#!/usr/bin/env bash
python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext '.mat' --name 'Phase02_test_001' --phase 'train' --save_latest_freq 28000 --print_freq 14000 --no_flip --batchSize 500 --checkpoints_dir './tests' --niter 75 --niter_decay 0 --quiet --normalize --magnitude --actvn 'relu' --cropped_signal --lr 0.0001 --se --pAct --G0 --plot_grads --dataset_mode 'LabeledMatSpectralDataset' --which_model_netEst 'resnet' --depth 1 --n_blocks 6 --use_dropout --parameters --metabolites --lambda_metab 1 --phase_data_path '/home/john/SpectroscopyModel/dataset'
python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext '.mat' --name 'Phase02_test_002' --phase 'train' --save_latest_freq 28000 --print_freq 14000 --no_flip --batchSize 500 --checkpoints_dir './tests' --niter 75 --niter_decay 0 --quiet --normalize --magnitude --actvn 'relu' --cropped_signal --lr 0.0001 --se --pAct --G0 --plot_grads --dataset_mode 'LabeledMatSpectralDataset' --which_model_netEst 'resnet' --depth 1 --n_blocks 6 --use_dropout --parameters --metabolites --lambda_metab 10 --phase_data_path '/home/john/SpectroscopyModel/dataset'
python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext '.mat' --name 'Phase02_test_003' --phase 'train' --save_latest_freq 28000 --print_freq 14000 --no_flip --batchSize 500 --checkpoints_dir './tests' --niter 75 --niter_decay 0 --quiet --normalize --magnitude --actvn 'relu' --cropped_signal --lr 0.0001 --se --pAct --G0 --plot_grads --dataset_mode 'LabeledMatSpectralDataset' --which_model_netEst 'lgresnet' --depth 1 --n_blocks 6 --use_dropout --parameters --metabolites --lambda_metab 1 --phase_data_path '/home/john/SpectroscopyModel/dataset' --lr_method 'cosine' --warmup 1
python train.py --dataroot '/home/john/SpectroscopyModel/dataset' --file_ext '.mat' --name 'Phase02_test_004' --phase 'train' --save_latest_freq 28000 --print_freq 14000 --no_flip --batchSize 500 --checkpoints_dir './tests' --niter 75 --niter_decay 0 --quiet --normalize --magnitude --actvn 'relu' --cropped_signal --lr 0.0001 --se --pAct --G0 --plot_grads --dataset_mode 'LabeledMatSpectralDataset' --which_model_netEst 'lgresnet' --depth 1 --n_blocks 6 --use_dropout --parameters --metabolites --lambda_metab 10 --phase_data_path '/home/john/SpectroscopyModel/dataset' --lr_method 'cosine' --warmup 1
