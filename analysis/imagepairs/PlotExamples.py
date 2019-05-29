'''
Script to plot ImagePair experiment generator examples.
Point the RESULTS_PATH to the folder where experiment logs are saved (that contains an "ImagePairs" folder)
The script will automatically plot generator examples for each experiment that is found in the folder.
The number of images to show for each model can be set by NUM_EXAMPLES.
NUM_COLS sets the number of columns (along horizontal axis) to use for plotting.
'''

import glob

import imageio as imageio
import numpy as np
import torch
import torchvision
import os

RESULTS_PATH = "/mnt/windaten/Results/factorGAN/"
DATASET_FOLDERS = ["cityscapes", "edges2shoes"]
OUT_PATH = ""
NUM_EXAMPLES = 16
NUM_COLS = 4

for dataset in DATASET_FOLDERS:
    for experiment_path in glob.glob(os.path.join(RESULTS_PATH, "ImagePairs", dataset, "*")):
        model = os.path.basename(experiment_path)
        gan_paths = [os.path.join(RESULTS_PATH, "ImagePairs", dataset, model, "gen", "gen_" + str(i) + ".png") for i in range(NUM_EXAMPLES)]
        gan_imgs = list()

        for file in gan_paths:
            gan_imgs.append(imageio.imread(file))
        gan_imgs = torch.from_numpy(np.transpose(np.stack(gan_imgs), [0, 3, 1, 2]))
        gan_imgs = torchvision.utils.make_grid(gan_imgs, nrow=NUM_COLS, padding=10, pad_value=255.0).permute(1, 2, 0)

        imageio.imwrite(os.path.join(OUT_PATH, "ImagePairs_" + dataset + "_" + model + "_gens.png"), gan_imgs)