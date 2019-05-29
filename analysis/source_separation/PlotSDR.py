'''
Plot SDR values for source separation experiment
'''

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.1, rc={'text.usetex' : True})

df = pd.read_csv("sdr.csv", delimiter=",")

g = sns.catplot("Songs", "Mean SDR", hue="Model", col="Source", data=df, kind="bar", height=3, aspect=2, sharey=False)

g.axes[0][0].set_title("Vocals")
g.axes[0][1].set_title("Accompaniment")

plt.savefig("sdr.pdf")