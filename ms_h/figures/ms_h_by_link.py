import matplotlib.pyplot as plt
import numpy as np
import glob, os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sb

main_fol = "F:\Hila\TDI\siemens"
exp = "D60d11"
all_subj_fol = glob.glob(f"{main_fol}{os.sep}{exp}{os.sep}[C,T]*{os.sep}")
atlas = "yeo7_100"
mat_type = "TDI_DistSampAvg"
num_of_nodes = 100
vec_size = num_of_nodes * (num_of_nodes - 1) // 2
indices = np.triu_indices(num_of_nodes, 1)
num_of_subjects = len(all_subj_fol)

group_label = np.zeros((num_of_subjects, 1))
values = np.zeros((num_of_subjects, vec_size))


for i, subj_fol in enumerate(all_subj_fol):
    if "tables" in subj_fol or "surfaces" in subj_fol:
        group_label[i] = np.nan
        values[i, :] = np.nan
        continue
    try:
        mat = np.load(f"{subj_fol}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy")
    except FileNotFoundError:
        print(f"couldn't find num_mat for {subj}")
        group_label[i] = np.nan
        values[i, :] = np.nan
        continue
    subj = subj_fol.split(os.sep)[-2]
    print(subj)
    if subj.startswith("C"):
        group_label[i] = 0
    elif subj.startswith("T0"):
        group_label[i] = 5
    elif subj.startswith("T1"):
        group_label[i] = 5
    elif subj.startswith("T2"):
        group_label[i] = np.nan
        values[i, :] = np.nan
    values[i, :] = mat[indices]

scaler = MinMaxScaler()
for i in range(vec_size):
    values[:, i] = scaler.fit_transform(values[:, i].reshape(-1, 1)).reshape(-1)

# draw swarmplot using different colors for each group labels where x is different edges and y is values:
df = pd.DataFrame(values)
df["group"] = group_label
df = pd.melt(df, id_vars=["group"])
df.columns = ["group", "edge", "value"]
df_mean = df.groupby(["group", "edge"], dropna=True).mean().reset_index()
plt.figure(figsize=(100, 4))
sb.stripplot(
    x="edge",
    y="value",
    hue="group",
    s=2,
    data=df_mean,
    jitter=False,
    palette="Spectral",
)
plt.show()
