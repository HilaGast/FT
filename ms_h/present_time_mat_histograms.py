import matplotlib.pyplot as plt
import numpy as np
import glob, os


main_fol = "F:\Hila\TDI\siemens"
exp = "D31d18"
all_subj_fol = glob.glob(f"{main_fol}{os.sep}{exp}{os.sep}[C,T]*{os.sep}")
atlas = "yeo7_100"
mat_type = "TDI_DistModeSym"
control_group_vals = []
ms_group_vals = []
medians = {}
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
        # plt.hist(mat[~np.isnan(mat)], bins=50, histtype='step', color='blue',linewidth=2, range=(0,100))
        # plt.imshow(mat, cmap="hot", vmin=0, vmax=100)
        # plt.colorbar()
        # plt.title(subj)
        # plt.show()

    elif group == "patient":
        ms_group_vals.extend(mat[~np.isnan(mat)])
        # plt.hist(mat[~np.isnan(mat)], bins=50, histtype='step', color='red',linewidth=2, range=(0,100))
        # plt.imshow(np.log(mat), cmap='jet', vmin=0, vmax=1)
        # plt.title(subj)
        # plt.show()
    medians[subj] = np.nanmedian(mat[mat > 0])
# plt.show()
control_group_vals = np.array(control_group_vals)
ms_group_vals = np.array(ms_group_vals)
range = (0, 250)
plt.hist(
    control_group_vals[control_group_vals > 0],
    bins=50,
    histtype="step",
    color="blue",
    linewidth=2,
    density=True,
    range=range,
)
plt.hist(
    ms_group_vals[ms_group_vals > 0],
    bins=50,
    histtype="step",
    color="red",
    linewidth=2,
    density=True,
    range=range,
)
plt.show()
