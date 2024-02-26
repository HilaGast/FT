import pandas as pd
import os, glob
import numpy as np

from HCP_network_analysis.group_avarage_mat import calc_group_average_mat
import matplotlib.pyplot as plt

main_fol = "G:\data\V7\HCP"
all_subj_fol = glob.glob(f"{main_fol}{os.sep}*[0-9]{os.sep}")
atlas_type = "yeo7_100"
mat_type = "time_th3"

genders = []
tdi = []
tdi_inter = []
tdi_intra = []
table1 = pd.read_csv("G:\data\V7\HCP\HCP_demographic_data.csv")

subj_f = []
subj_m = []


for subj in all_subj_fol:
    subj_number = subj.split(os.sep)[-2]
    gender = table1["Gender"][table1["Subject"] == int(subj_number)].values
    genders.append(gender)
    if gender == "F":
        subj_f.append(
            f"{subj}{os.sep}cm{os.sep}{atlas_type}_{mat_type}_Org_SC_cm_ord.npy"
        )
    elif gender == "M":
        subj_m.append(
            f"{subj}{os.sep}cm{os.sep}{atlas_type}_{mat_type}_Org_SC_cm_ord.npy"
        )

mean_mat = calc_group_average_mat(subj_f, atlas_type, type="median")
np.save(
    f"{main_fol}{os.sep}cm{os.sep}median_{atlas_type}_{mat_type}_Org_SC_female.npy",
    mean_mat,
)
mean_mat = calc_group_average_mat(subj_m, atlas_type, type="median")
np.save(
    f"{main_fol}{os.sep}cm{os.sep}median_{atlas_type}_{mat_type}_Org_SC_male.npy",
    mean_mat,
)

f = np.load(
    rf"{main_fol}{os.sep}cm{os.sep}median_{atlas_type}_{mat_type}_Org_SC_female.npy"
)
m = np.load(
    rf"{main_fol}{os.sep}cm{os.sep}median_{atlas_type}_{mat_type}_Org_SC_male.npy"
)
plt.hist(f[f > 0], bins=50, color="blue", alpha=0.2, range=(0, 500), density=True)
plt.hist(m[m > 0], bins=50, color="green", alpha=0.2, range=(0, 500), density=True)
plt.hist(
    f[f > 0],
    bins=50,
    histtype="step",
    color="blue",
    linewidth=2,
    range=(0, 500),
    density=True,
)
plt.hist(
    m[m > 0],
    bins=50,
    histtype="step",
    color="green",
    linewidth=2,
    range=(0, 500),
    density=True,
)
plt.legend(["Females", "Males"])
plt.show()
