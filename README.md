# Overview

This respository implements "FactorGAN" as described in the paper

"Training Generative Adversarial Networks from Incomplete Observations using Factorised Discriminators"

## FactorGAN - Quick introduction

Consider training a GAN to solve some generation task, where a sample can be naturally divided into multiple parts.
As one example, we will use images of shoes along with a drawing of their edges (see diagram below).

With a normal GAN for this task, the generator just outputs a pair of images, which the discriminator then analyses as a whole.
For training the discriminator, you use real shoe-edge image pairs from a "paired" dataset.

But what if you only have a few of these paired samples, but many more individual shoe OR edge images?
You cannot use these for GAN training to improve the quality of your shoes and edges further, as the discriminator needs shoe-edge image pairs.

That is where the FactorGAN comes in. As shown below, it makes use of all available data (shoe-edge pairs, shoes alone, edges alone):

<img src="./factorgan_unconditional.png" width="650">

To achieve this, FactorGAN uses four discriminators:
* a discriminator to judge the generator's shoe quality
* another to do the same for the edges
* two "dependency discriminators" to ensure the generator outputs edge maps that fit to their respective shoe images: 
    * The "real dependency" discriminator tries to distinguish real paired examples from real ones where each shoe was randomly assigned to an edge map (by "shuffling" the real batch), thereby having to learn which edge maps go along which shoes.
    * The "fake dependency" discriminator does the same for generator samples.
The real dependency discriminator is the only component that needs paired samples for training, while the other components can make use of the extra available shoes and edge images.

Training works by alternating between a) updating all discriminators (individually) and b) updating the generator.
Amazingly, we can update the generator just like in a normal GAN, by simply adding the unnormalised discriminator outputs and using the result for the generator loss.
This combined output can be proven to approximate the same probability for real and fake inputs as estimated by a normal GAN discriminator.

In our experiments, the FactorGAN provides very good output quality even with just very few paired samples, as the shoe and edge discriminators trained on the additional unpaired samples help the generator to output realistic shoes and edge maps. 

This principle can also be used for conditional generation (aka prediction tasks).
Let's take image segmentation as an example:

<img src="./factorgan_conditional.png" width="750">

The generator now acts as a segmentation model, predicting the segmentation from a given city scene.
In contrast to a normal conditional GAN, whose discriminator requires the scene along with its segmentation as "paired" input, here we use
* a discriminator acting only on real and fake segmentations, trainable with individual scenes and segmentation maps, ensuring the predicted segmentation is "realistic" on its own, irrespective of the scene it was predicted from
* a fake dependency discriminator that distinguishes (real scene, fake segmentation) pairs from their shuffled variant, to learn how the generator output corresponds to the input
* a real dependency discriminator that distinguishes (real scene, real segmentation) pairs from their shuffled variants.

We perform segmentaiton experiments on the Cityscapes dataset, treating the samples as unpaired (like the CycleGAN).
But we find that adding as few as 25 paired samples yields substantially higher segmentation accuracy than the CycleGAN - suggesting that the FactorGAN fills a gap between fully unsupervised and fully supervised methods by making efficient use of both paired and unpaired samples.   

# Requirements

* Python 3.6
* Pip for installing Python packages
* [libsnd](http://www.mega-nerd.com/libsndfile/) library installed
* [wget](https://www.gnu.org/software/wget/) installed for downloading the datasets automatically
* GPU is optional, but strongly recommended to avoid long computation times 

# Installation

Install the required packages as listed in ``requirements.txt``.
To ensure existing packages do not interfere with the installation, it is best to create a virtual environment with ``virtualenv`` first and then install the packages separately into that environment. 
Easy installation can be performed using 
```
pip install -r requirements.txt
``` 

## Dataset download

### Cityscapes and Edges2Shoes

For experiments involving Cityscapes or Edges2Shoes data, you need to download these datasets first. 
To do this, change to the ``datasets/image2image`` subfolder in your commandline using

```
cd datasets/image2image
```

and then simply execute

```
./download_image2image.sh cityscapes
```

or

```
./download_image2image.sh edges2shoes
```

### MUSDB18 (audio separation)

For audio source separation experiments, you will need to download the [MUSDB18 dataset](https://sigsep.github.io/datasets/musdb.html) from [Zenodo](https://zenodo.org/record/1117372) manually, since it requires requesting access, and extract it to a folder of your choice.

When running the training script, you can point to the MUSDB dataset folder by giving its path as a command-line parameter.

# Running experiments

To run the experiments, execute the script corresponding to the particular application, from the root directory of the repository:
* ```PairedMNIST.py```: Paired MNIST experiments
* ```ImagePairs.py```: Generation of image pairs (Cityscapes, Edges2Shoes)
* ```Image2Image.py```: Used for image segmentation (Cityscapes)
* ```AudioSeparation.py```: For vocal separation

Each experiment in the paper can be replicated by specifying the experimental parameters via the commandline.

Firstly, there is a set of parameters shared between all experiments, which are described in ```training/TrainingOptions.py```.
The most important ones are:
* ```--cuda```: Activate GPU training flag
* ```--experiment_name```: Provide a string to name the experiment, which will be used to name the output folder
* ```--out_path```: Provide the output folder where results and logs are saved. All output will usually be in ```out_path/TASKNAME/experiment_name```.
* ```--eval```: Append this flag if you only want to perform model evaluation for an already trained model. CAUTION: ``experiment_name`` path as well as network parameters have to be set correctly (like the one used during training) to ensure this works correctly.
* ```--factorGAN```: Provide a 0 to use the normal GAN, 1 for FactorGAN
* ```--use_real_dep_disc```:  Provide a 0 to not use a p-dependency discriminator, 1 for the full FactorGAN

Every experiment also has specific extra commandline parameters which are explained in the code file for each experiment.

## Examples

Train a GAN on PairedMNIST with 1000 joint samples, using GPU, and save results in ```out/PairedMNIST/100_samples_GAN```:  
```
python PairedMNIST.py --cuda --factorGAN 0 --num_joint_samples 1000 --experiment_name "100_samples_GAN"
```

Train a FactorGAN to generate scene-segmentation image pairs with 100 joint samples on the Cityscapes dataset, using GPU, and save results in ```out/ImagePairs/cityscapes/100_samples_factorGAN```:  
```
python ImagePairs.py --cuda --dataset "cityscapes" --num_joint_samples 100 --factorGAN 1 --experiment_name "100_samples_factorGAN"
```

Train a FactorGAN to for image segmentation on the Cityscapes dataset with 25 joint samples on the Cityscapes dataset, using GPU, and save results in ```out/Image2Image_cityscapes/25_samples_factorGAN```:  
```
python Image2Image.py --cuda --dataset "cityscapes" --num_joint_samples 25 --factorGAN 1 --experiment_name "25_samples_factorGAN"
```

# Analysing and plotting results

The ```analysis``` subfolder contains 
* some of the results obtained during our experiments, such as performance metrics
* all scripts that were used to produce the figures
* the figures used in the paper

Some of them require the full output of an experiment, so the experiment needs to be run first, and the path to the resulting output folder can be inserted into the script. They are not included in the repository directly as they can be quite large.
