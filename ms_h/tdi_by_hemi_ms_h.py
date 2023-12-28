import os, glob
import numpy as np
from ms_h.present_time_mat_by_hemisphere import divide_mat_to_inter_intra_hemi_mats

main_fol = 'F:\Hila\TDI\siemens'
exp = 'D31d18'
atlas = 'yeo7_100'
mat_type = 'TDI_EucSym'
#mat_type = "EucDist"
h_mat_files = glob.glob(f'{main_fol}{os.sep}{exp}{os.sep}C*{os.sep}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy')
ms_mat_files = glob.glob(f'{main_fol}{os.sep}{exp}{os.sep}T*{os.sep}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy')

h_wb = []
h_inter = []
h_intra = []
ms_wb = []
ms_inter = []
ms_intra = []

for h_mat_file in h_mat_files:
    mat = np.load(h_mat_file)
    mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat, atlas)
    mat[mat == 0] = np.nan
    h_wb.append(np.nanmean(mat))
    mat_intra[mat_intra == 0] = np.nan
    h_intra.append(np.nanmean(mat_intra))
    mat_inter[mat_inter == 0] = np.nan
    h_inter.append(np.nanmean(mat_inter))

for ms_mat_file in ms_mat_files:
    mat = np.load(ms_mat_file)
    mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat, atlas)
    mat[mat == 0] = np.nan
    ms_wb.append(np.nanmean(mat))
    mat_intra[mat_intra == 0] = np.nan
    ms_intra.append(np.nanmean(mat_intra))
    mat_inter[mat_inter == 0] = np.nan
    ms_inter.append(np.nanmean(mat_inter))

print(f'WB: H: {np.nanmean(h_wb)}, MS: {np.nanmean(ms_wb)}')
print(f'Intra: H: {np.nanmean(h_intra)}, MS: {np.nanmean(ms_intra)}')
print(f'Inter: H: {np.nanmean(h_inter)}, MS: {np.nanmean(ms_inter)}')

import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

sb.boxplot([h_wb, ms_wb,h_intra, ms_intra,h_inter, ms_inter], palette=['blue', 'red','blue', 'red','blue', 'red'], width = 0.3)
plt.xticks([0,1, 2, 3, 4, 5],['WB H', 'WB MS','Intra H', 'Intra MS','Inter H', 'Inter MS'])
plt.title(exp)
plt.show()

t,p = ttest_ind(h_wb, ms_wb,alternative='less')
print(f'Whole brain: t: {t}, p: {p}')

t,p = ttest_ind(h_intra, ms_intra,alternative='less')
print(f'Intra-hemisphere: t: {t}, p: {p}')

t, p = ttest_ind(h_inter, ms_inter,alternative='less')
print(f'Inter-hemisphere: t: {t}, p: {p}')