# Generate simulated datasets
# Syn_real
echo Generating Syn_real...
python generate_spectra.py --save_path "datasets/syn_real.mat" --N 100000
# Syn_ucsf
echo Generating Syn_ucsf...
python generate_spectra.py --save_path "datasets/syn_ucsf.mat" --quantitity_path "MRSI_data.mat"
# Syn_ideal
echo Generating Syn_ideal...
python generate_spectra.py --save_path "datasets/syn_ideal.mat" --N 100000 --β_min 1.0 --β_max 1.0 --SNR_min 100 --SNR_max 100


# Create baslines and pretrain models
echo Creating baselines and pretrain models...
python create_baseline.py
echo Done