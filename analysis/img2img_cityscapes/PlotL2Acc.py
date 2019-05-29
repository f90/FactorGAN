'''
Plot L2 and MSE accuracy of image segmentation models (GAN vs FactorGAN)
'''

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(font_scale=1.1, rc={'text.usetex' : True})

df = pd.read_csv("Image2Image_Cityscapes.csv", delimiter=",")
g = sns.catplot("Paired samples", "Perf", hue="Model", col="Metric", data=df, kind="bar", sharey=False, height=3, aspect=2)
g.axes[0][0].set_ylabel("Mean squared error")
g.axes[0][0].set_title("")
g.axes[0][1].set_title("")
g.axes[0][1].set_ylabel("Accuracy (\%)")

g.fig.subplots_adjust(top=0.95, wspace=0.15)

plt.savefig("cityscapes.pdf")