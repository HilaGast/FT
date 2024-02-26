import matplotlib.pyplot as plt
import numpy as np
import glob, os


main_fol = "F:\Hila\TDI\siemens"
exp = "D31d18"
all_subj_fol = glob.glob(f"{main_fol}{os.sep}{exp}{os.sep}[C,T]*{os.sep}")
atlas = "yeo7_100"
mat_type = "TDI_DistSampAvg"
control_group_vals = []
ms0_group_vals = []
ms1_group_vals = []
ms2_group_vals = []


for subj_fol in all_subj_fol:
    if "tables" in subj_fol or "surfaces" in subj_fol:
        continue
    subj = subj_fol.split(os.sep)[-2]
    print(subj)
    if subj.startswith("C"):
        group = "control"
    else:
        group = "patient"
    try:
        mat = np.load(f"{subj_fol}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy")
    except FileNotFoundError:
        print(f"couldn't find num_mat for {subj}")
        continue
    # mat = 100 * mat / np.nansum(mat)
    if group == "control":
        control_group_vals.extend(mat[~np.isnan(mat)])

    elif group == "patient":
        if subj.startswith("T0"):
            ms0_group_vals.extend(mat[~np.isnan(mat)])
        elif subj.startswith("T1"):
            ms1_group_vals.extend(mat[~np.isnan(mat)])
        elif subj.startswith("T2"):
            ms2_group_vals.extend(mat[~np.isnan(mat)])


# plt.show()
control_group_vals = np.array(control_group_vals)
ms0_group_vals = np.array(ms0_group_vals)
ms1_group_vals = np.array(ms1_group_vals)
ms2_group_vals = np.array(ms2_group_vals)

range = (1, 100)
plt.hist(
    control_group_vals[control_group_vals > 0],
    bins=30,
    histtype="step",
    color="blue",
    linewidth=2,
    density=True,
    range=range,
)
plt.hist(
    ms0_group_vals[ms0_group_vals > 0],
    bins=30,
    histtype="step",
    color="red",
    linewidth=2,
    density=True,
    range=range,
)
plt.legend(["Control", "MS0"])
plt.title(f"Median values of {mat_type} \n {atlas} atlas - {exp}")
plt.show()

plt.hist(
    control_group_vals[control_group_vals > 0],
    bins=30,
    histtype="step",
    color="blue",
    linewidth=2,
    density=True,
    range=range,
)
plt.hist(
    ms1_group_vals[ms1_group_vals > 0],
    bins=30,
    histtype="step",
    color="red",
    linewidth=2,
    density=True,
    range=range,
)
plt.legend(["Control", "MS1"])
plt.title(f"Median values of {mat_type} \n {atlas} atlas - {exp}")
plt.show()

plt.hist(
    control_group_vals[control_group_vals > 0],
    bins=30,
    histtype="step",
    color="blue",
    linewidth=2,
    density=True,
    range=range,
)
plt.hist(
    ms2_group_vals[ms2_group_vals > 0],
    bins=30,
    histtype="step",
    color="red",
    linewidth=2,
    density=True,
    range=range,
)
plt.legend(["Control", "MS2"])
plt.title(f"Median values of {mat_type} \n {atlas} atlas - {exp}")
plt.show()
