import pandas as pd
import os, glob
import numpy as np

from figure_creation_scripts.weights_correlations import draw_scatter_fit_weights
from ms_h.figures.present_time_mat_by_hemisphere import (
    divide_mat_to_inter_intra_hemi_mats,
)

table_years = pd.read_excel(r"F:\Hila\TDI\TheBase4Ever subjects.xlsx")
atlas_type = "yeo7_200"
mat_type = "time_binary_EucDist"
ages = []
tdi = []
tdi_intra = []
tdi_inter = []
main_fol = "F:\Hila\TDI\TheBase4Ever"
all_subj_fol = glob.glob(f"{main_fol}{os.sep}[0-9]*{os.sep}")

for subj in all_subj_fol:
    subj_name = subj.split(os.sep)[-2]
    subj_index = table_years["Scan File 1"].str.contains(subj_name)
    try:
        age = table_years["Age"][subj_index].values[0]
    except IndexError:
        subj_index = table_years["Scan File 2"].str.contains(subj_name)
        subj_index = subj_index[subj_index == True].index[0]
        age = table_years["Age"][subj_index]
    if 18 < age < 80:
        ages.append(age)
        mat = np.load(f"{subj}cm{os.sep}{atlas_type}_{mat_type}_cm_ord.npy")
        mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat, atlas_type)
        mat[mat == 0] = np.nan
        mat_intra[mat_intra == 0] = np.nan
        mat_inter[mat_inter == 0] = np.nan
        tdi.append(np.nanmedian(mat))
        tdi_intra.append(np.nanmedian(mat_intra))
        tdi_inter.append(np.nanmedian(mat_inter))
    else:
        continue

# draw_scatter_fit_weights(ages, tdi, 'Age', 'Euclidean Distance',(20,75),(30,60), c=[0.3,0.3,0.5])
# draw_scatter_fit_weights(ages, tdi_intra, 'Age', 'Euclidean Distance \n(inside hemispheres)',(20,75),(30,60), c=[0.3,0.3,0.5])
# draw_scatter_fit_weights(ages, tdi_inter, 'Age', 'Euclidean Distance \n(between hemispheres)',(20,75),(30,60), c=[0.3,0.3,0.5])
#
draw_scatter_fit_weights(
    ages, tdi, "Age", "TDI", (18, 80), (10, 150), c=[0.3, 0.3, 0.5]
)
draw_scatter_fit_weights(
    ages,
    tdi_intra,
    "Age",
    "TDI \n(inside hemispheres)",
    (18, 80),
    (10, 150),
    c=[0.3, 0.3, 0.5],
)
draw_scatter_fit_weights(
    ages,
    tdi_inter,
    "Age",
    "TDI \n(between hemispheres)",
    (18, 80),
    (10, 150),
    c=[0.3, 0.3, 0.5],
)
