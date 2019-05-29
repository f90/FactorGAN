'''
Plot d_dep dependency metric for PairedMNIST experiment
'''

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(font_scale=1.1, rc={'text.usetex' : True})

fig, ax = plt.subplots(figsize=(6,3.5))

df = pd.read_csv("PairedMNIST Results Diff.csv", delimiter=";")

# Get no-cp performances
low_lambda_nodep = df[(df["Model"] == "FactorGAN-no-cp") & (df["Lambda"] == 0.1)]["Diff"].as_matrix()[0]
high_lambda_nodep = df[(df["Model"] == "FactorGAN-no-cp") & (df["Lambda"] == 0.9)]["Diff"].as_matrix()[0]

# Filter facgan-no-cp
df = df[df["Model"] != "FactorGAN-no-cp"]

df["Model"] = df["Model"] + ", $\lambda$=" + df["Lambda"].apply(lambda x: str(x))

ax.axhline(y=low_lambda_nodep, c='black', linestyle='--', label="FactorGAN-no-cp, $\lambda=0.1$", alpha=0.8)
ax.axhline(y=high_lambda_nodep, c='gray', linestyle='--', label="FactorGAN-no-cp, $\lambda=0.9$", alpha=0.8)

sns.barplot("Paired samples", "Diff", hue="Model", data=df, ax=ax)

ax.set_ylabel("Dependency metric $d_{dep}$")

# Sort legend
handles, labels = ax.get_legend_handles_labels()
handles = handles[2:] + handles[0:2]
labels = labels[2:] + labels[0:2]
ax.legend(handles, labels)

#plt.legend()
fig.tight_layout()
fig.savefig("mnist_dep.pdf")