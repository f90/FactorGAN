'''
Plot FID values for PairedMNIST experiment
'''

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(font_scale=1.1, rc={'text.usetex' : True})

fig, ax = plt.subplots(figsize=(6,3.5))
# PLOT FID
df = pd.read_csv("PairedMNIST Results FID.csv", delimiter=";")

# Get no-cp performances
low_lambda_nodep = df[(df["Model"] == "FactorGAN-no-cp") & (df["Lambda"] == 0.1)]["FID"].as_matrix()[0]
high_lambda_nodep = df[(df["Model"] == "FactorGAN-no-cp") & (df["Lambda"] == 0.9)]["FID"].as_matrix()[0]

# Filter facgan-no-cp
df = df[df["Model"] != "FactorGAN-no-cp"]

# Combine model with lambda
df["Model"] = df["Model"] + ", $\lambda$=" + df["Lambda"].apply(lambda x: str(x))

sns.barplot("Paired samples", "FID", hue="Model", data=df, ax=ax)
ax.set_yscale("log")

ax.axhline(y=low_lambda_nodep, c='black', linestyle='dashed', label="FactorGAN-no-cp, $\lambda=0.1$", alpha=0.8)
ax.axhline(y=high_lambda_nodep, c='gray', linestyle='dashed', label="FactorGAN-no-cp, $\lambda=0.9$", alpha=0.8)

# PLOT
# Sort legend
handles, labels = ax.get_legend_handles_labels()
handles = handles[2:] + handles[0:2]
labels = labels[2:] + labels[0:2]
ax.legend(handles, labels) # bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

#ax.get_legend().remove()
fig.tight_layout()
fig.savefig("mnist_fid.pdf")