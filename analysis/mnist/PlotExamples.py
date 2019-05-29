'''
Plot PairedMNIST generated examples
'''

import os

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

fig, axes = plt.subplots(2, 3)
root_path = "../../out/PairedMNIST"

for idx, samples in enumerate([100, 5000, 20000]):
    for model_idx, model in enumerate(["GAN", "factorGAN"]):
        gan_paths = [os.path.join(root_path, str(samples) + "_joint_0.9_samedigit_" + model, "gen", "gen_" + str(i) + ".png") for i in range(28)]

        gan_imgs = list()
        for file in gan_paths:
            gan_imgs.append(imageio.imread(file))
        gan_imgs = torch.from_numpy(np.transpose(np.stack(gan_imgs), [0, 3, 1, 2]))
        gan_imgs = torchvision.utils.make_grid(gan_imgs, nrow=7, padding=5, pad_value=255.0).permute(1, 2, 0)

        axes[model_idx][idx].imshow(gan_imgs)
        axes[model_idx][idx].axis("off")

plt.subplots_adjust(wspace=0.1, hspace=.1)

axes[0][0].text(-0.3,0.5, "GAN", size=12, ha="center", transform=axes[0][0].transAxes)
axes[1][0].text(-0.3,0.5, "factorGAN", size=12, ha="center", transform=axes[1][0].transAxes)

axes[1][0].text(0.5,-0.1, "100", size=12, ha="center", transform=axes[1][0].transAxes)
axes[1][1].text(0.5,-0.1, "500", size=12, ha="center", transform=axes[1][1].transAxes)
axes[1][2].text(0.5,-0.1, "20000", size=12, ha="center", transform=axes[1][2].transAxes)

plt.savefig("mnist_examples.pdf", bbox_inches="tight")