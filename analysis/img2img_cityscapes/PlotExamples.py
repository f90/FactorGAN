'''
Visualise Cityscapes image segmentation predictions. Requires experiments folders to be existing with the correct names first!
'''
import glob

import imageio as imageio
import numpy as np
import torch
import torchvision
import os

NUM_ROWS = 3
NUM_COLUMNS = 2

BASEDIR = "../../out/Image2Image_cityscapes/"
EXPERIMENTS = [os.path.basename(p) for p in glob.glob(os.path.join(BASEDIR, "*"))]

for experiment in EXPERIMENTS:
    gan_paths = [os.path.join(BASEDIR, experiment, "gen", "gen_" + str(i) + ".png") for i in range(NUM_ROWS*NUM_COLUMNS)]
    gan_imgs = list()

    for file in gan_paths:
        gan_imgs.append(imageio.imread(file))
    gan_imgs = torch.from_numpy(np.transpose(np.stack(gan_imgs), [0, 3, 1, 2]))
    gan_imgs = torchvision.utils.make_grid(gan_imgs, nrow=2, padding=10, pad_value=255.0).permute(1, 2, 0)

    imageio.imwrite("cityscapes_" + experiment + "_gens.png", gan_imgs)