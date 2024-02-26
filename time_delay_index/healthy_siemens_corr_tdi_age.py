import pandas as pd
import os, glob
import numpy as np
import matplotlib.pyplot as plt

from ms_h.figures.average_time_mat_by_group import average_time_mat

main_fol = "F:\Hila\TDI\siemens"
experiments = ["D60d11", "D45d13", "D31d18"]
table_years = pd.read_excel(r"F:\Hila\TDI\siemens\age_and_duration.xlsx")
atlas_type = "bnacor"
mat_type = "time_th3"

for exp in experiments:
    ages = []
    # tdi = []

    subj_y = []
    subj_m = []
    subj_o = []
    h_subj_fol = glob.glob(f"{main_fol}{os.sep}C*{os.sep}{exp}")
    for subj in h_subj_fol:
        subj_name = subj.split(os.sep)[-2]
        age = table_years["Age"][table_years["subj"] == subj_name].values[0]
        ages.append(age)
        if age < 30:
            subj_y.append(subj)
        elif age > 40:
            subj_o.append(subj)
        else:
            subj_m.append(subj)
        # mat = np.load(f'{subj}{os.sep}cm{os.sep}{mat_type}_{atlas_type}_cm_ord.npy')
        # mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat, 'bnacor')
        # mat[mat==0] = np.nan
        # tdi.append(np.nanmedian(mat))

    # r, p = calc_corr(ages, tdi)
    # print(f'WB: {exp} - r:{r} - p:{p}')
    # plt.plot(ages, tdi, 'o')
    # plt.title('WB')
    # plt.show()
    average_time_mat(subj_y, f"{exp}_young", main_fol, mat_type, atlas_type)
    average_time_mat(subj_m, f"{exp}_middle", main_fol, mat_type, atlas_type)
    average_time_mat(subj_o, f"{exp}_old", main_fol, mat_type, atlas_type)

    a = np.load(rf"F:\Hila\TDI\siemens\median_{mat_type}_{atlas_type}_{exp}_young.npy")
    b = np.load(rf"F:\Hila\TDI\siemens\median_{mat_type}_{atlas_type}_{exp}_middle.npy")
    c = np.load(rf"F:\Hila\TDI\siemens\median_{mat_type}_{atlas_type}_{exp}_old.npy")
    plt.hist(a[a > 0], bins=50, color="blue", alpha=0.2, range=(0, 500), density=True)
    plt.hist(b[b > 0], bins=50, color="red", alpha=0.2, range=(0, 500), density=True)
    plt.hist(c[c > 0], bins=50, color="green", alpha=0.2, range=(0, 500), density=True)
    plt.hist(
        a[a > 0],
        bins=50,
        histtype="step",
        color="blue",
        linewidth=2,
        range=(0, 500),
        density=True,
    )
    plt.hist(
        b[b > 0],
        bins=50,
        histtype="step",
        color="red",
        linewidth=2,
        range=(0, 500),
        density=True,
    )
    plt.hist(
        c[c > 0],
        bins=50,
        histtype="step",
        color="green",
        linewidth=2,
        range=(0, 500),
        density=True,
    )
    plt.legend(["Young (21-30)", "Middle (30-40)", "Old (>40)"])
    plt.show()
