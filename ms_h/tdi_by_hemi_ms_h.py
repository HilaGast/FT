import os, glob
import numpy as np
from ms_h.present_time_mat_by_hemisphere import divide_mat_to_inter_intra_hemi_mats

main_fol = 'F:\Hila\TDI\siemens'
exp = 'D31d18'
atlas = 'bnacor'
mat_type = 'time_th3'
h_mat_files = glob.glob(f'{main_fol}{os.sep}C*{os.sep}{exp}{os.sep}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy')
ms_mat_files = glob.glob(f'{main_fol}{os.sep}T*{os.sep}{exp}{os.sep}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy')

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

print(f'WB: H: {np.mean(h_wb)}, MS: {np.mean(ms_wb)}')
print(f'Intra: H: {np.mean(h_intra)}, MS: {np.mean(ms_intra)}')
print(f'Inter: H: {np.mean(h_inter)}, MS: {np.mean(ms_inter)}')

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