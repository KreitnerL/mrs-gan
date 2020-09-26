# MRS-GAN

Using a Generative Adversarial Network (GAN) to decrease the domain shift between real and synthtic Magnetic Resonance Spectroscopy (MRS) samples. Code base for Bachelor Thesis.

## Introduction:
Magnetic Resonance Spectroscopy (MRS) is an analytical technique to detect metabolic changes in brain tumors. It complemenets the better known Magnetic Resonance Imaging (MRI) scans to further characterise the examined tissue.

![MRS and MRI sample](docs/images/MRSI_sample.png)
*Figure 1: MRS and MRI sample*

*The graph shows the concentration of various biochemicals (metabolites) in the examined tissue. This spectra can be used to make assumptions about the nature of a possible tumor. The MRI scan (three images on the upper right) gives imformation about the physical structure of the brain and the tumor. Both methods work complementary.*

 Even with todays methods it is still hard to determine certain rules when it comes to regression tasks of MRS samples. The ways of interpretation are too manifold and they are prone to even slight changes of the spectra. Because of this dilemma, the idea of using a machine learning approach for spectral quantification was born. However, to train a neural network, millions of labeled sample data would need to be required. The aquisition of such samples is extremly ressourceful and time consuming. Furthermore, ground truth metabolite concentrations are not available for *in vivo* signals, even by using medical experts. Therefore, it might be easier to generate synthetic sample data in order to train the classifier. 
However, when we switch from synthetic data to real data we are facing a domain shift. This is partly because every MRS scanner produces individual non-linearities, resulting in different looking spectras. The pre-trained network does not yet adapt to this domain shift and thus yields inferior results.
To overcome this domain gap we have two options:

1) We could find a way of improving the quality of the synthetic data to better match the real data. We would then train the network on this now more realistic looking data to decrease the domain difference. However, as stated before, each scanner produces a different distribution, hence it is almost impossible to find a representation that matches well with all scanners. Therefore, one would need to train not only the transforming network specifically for each scanner, but also the regression network, since it has to learn to handle samples from this domain.
2) We could train a network to transform the spectra of a scanner to look more like the clean systhetic data the regression network encountered in training. For this, we would only need to retrain the transforming network for each scanner and can feed the result into the pre-trained regression network. 

This project is an attempt to build a transformer network that can be used for both use cases. For this, we use a version of CycleGAN than uses two  distinct Generator networks to transform signals from domain A to domain B or from B to A respectively.

![CycleGAN](docs/images/cyclegan.png)
![CycleGAN](docs/images/real-gen-rec.png)
*Figure 2: Idea of CycleGAN*

*Given samples from domain D_x, a generator network G can transform the sample to a target domain D_y. The second generator F can transform the image back to its original domain. The cycle consistency loss ensures that the structure of the image is preserved.*

## How to Use:
### Prerequesites:
Before you begin, make sure you have a valid python3 installation, and a CUDA-enabled GPU (https://developer.nvidia.com/cuda-gpus).

First, install all the necessary requirements:
```sh
pip install -r requirements.txt
```

Download a CycleGAN dataset (e.g. vangogh2photo):
```sh
bash ./datasets/download_cyclegan_dataset.sh vangogh2photo
```

### Training:
You can train the model with:
```sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --no_dropout
```
Check the [options folder](./options/README.md) for more information about the run parameters.


### Testing:
To test the pre-trained model, run:
```sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --phase test --no_dropout
```

The test results will be saved to a html file here: ./results/maps_cyclegan/latest_test/index.html
