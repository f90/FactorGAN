import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(font_scale=1.1, rc={'text.usetex' : True})

df = pd.read_csv("ImagePairs Results.csv", delimiter=",")
df.loc[df["Paired samples"] > 1000, "Paired samples"] = "All"

g = sns.catplot(x="Paired samples", y="LS", hue="Model", col="Dataset", data=df, kind="bar", ci=95, height=3, aspect=2)#, hue_order=["FactorGAN", "GAN", "GAN (big)"])

g.axes[0][0].set_title("Edges2Shoes")
g.axes[0][1].set_title("Cityscapes")

#plt.tight_layout()
plt.savefig("imagepairs_LS.pdf")