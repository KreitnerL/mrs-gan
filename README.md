# MRS-GAN

Using a Generative Adversarial Network (GAN) to create realistic Magnetic Resonance Spectroscopy (MRS) samples. Code base for Bachelor Thesis.

## Introduction
Magnetic Resonance Spectroscopy (MRS) is an analytical technique to detect metabolic changes in brain tumors. It complemenets the better known Magnetic Resonance Imaging (MRI) scans to further characterise the examined tissue. Even with todays methods it is still hard to determine certain rules when it comes to classification of MRS samples. The ways of interpretation are too manifold and they are prone to even slight changes of the spectra. Because of this dilemma, the idea of using a machine learning approach was born. However, to train a neural network, millions of labeled sample data would need to be required. The aquisition of such samples is extremly ressourceful and time consuming. Therefore, it might be easier to generate synthetic sample data in order to train the classifier. Simple generated data samples do not yet yield satisfying results, hence this project aims to improve the quality of existing synthic MRS samples to overcome the domain shift from synthetic to real data. 

![Image of Yaktocat](docs/images/MRSI_sample.png)
*Figure 1: MRS(I) sample*
*The graph shows the concentration of various biochemicals (metabolites) in the examined tissue. This spectra can be used to make assumptions about the nature of a possible tumor. The MRI scan (three images on the upper right) gives imformation about the physical structure of the brain and the tumor. Both methods work complementary.*

## How to Use
Before you begin, make sure you have a valid python3 installation, a CUDA-enabled GPU (https://developer.nvidia.com/cuda-gpus).

First, install all the necessary requirements:
```sh
pip install -r requirements.txt
```

You can train the model with:
```sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --no_dropout
```

TODO
