import os, glob
import numpy as np
from ms_h.present_time_mat_by_hemisphere import divide_mat_to_inter_intra_hemi_mats
import pandas as pd

main_fol = 'F:\Hila\TDI\siemens'
experiments = ['D60d11', 'D45d13', 'D31d18']
table_years = pd.read_excel(r'F:\Hila\TDI\siemens\age_and_duration.xlsx')
exp = experiments[0]
atlas = 'bnacor'
mat_type = 'time_th3'

subj_y = []
subj_m = []
subj_o = []
h_subj_fol = glob.glob(f'{main_fol}{os.sep}C*{os.sep}{exp}')
for subj in h_subj_fol:
    subj_name = subj.split(os.sep)[-2]
    age = table_years['Age'][table_years['subj'] == subj_name].values[0]

    if age < 30:
        subj_y.append(subj)
    elif age > 40:
        subj_o.append(subj)
    else:
        subj_m.append(subj)
y_wb = []
y_intra = []
y_inter = []
m_wb = []
m_intra = []
m_inter = []
o_wb = []
o_intra = []
o_inter = []

for y_mat_file in subj_y:
    mat_name = f'{y_mat_file}{os.sep}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy'
    mat = np.load(mat_name)
    mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat, atlas)
    mat[mat == 0] = np.nan
    y_wb.append(np.nanmean(mat))
    mat_intra[mat_intra == 0] = np.nan
    y_intra.append(np.nanmean(mat_intra))
    mat_inter[mat_inter == 0] = np.nan
    y_inter.append(np.nanmean(mat_inter))

for m_mat_file in subj_m:
    mat_name = f'{m_mat_file}{os.sep}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy'
    mat = np.load(mat_name)
    mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat, atlas)
    mat[mat == 0] = np.nan
    m_wb.append(np.nanmean(mat))
    mat_intra[mat_intra == 0] = np.nan
    m_intra.append(np.nanmean(mat_intra))
    mat_inter[mat_inter == 0] = np.nan
    m_inter.append(np.nanmean(mat_inter))

for o_mat_file in subj_o:
    mat_name = f'{o_mat_file}{os.sep}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy'
    mat = np.load(mat_name)
    mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat, atlas)
    mat[mat == 0] = np.nan
    o_wb.append(np.nanmean(mat))
    mat_intra[mat_intra == 0] = np.nan
    o_intra.append(np.nanmean(mat_intra))
    mat_inter[mat_inter == 0] = np.nan
    o_inter.append(np.nanmean(mat_inter))


print(f'WB: young: {np.mean(y_wb)}, middle: {np.mean(m_wb)}, old: {np.mean(o_wb)}')
print(f'Intra: young: {np.mean(y_intra)}, middle: {np.mean(m_intra)}, old: {np.mean(o_intra)}')
print(f'Inter: young: {np.mean(y_inter)}, middle: {np.mean(m_inter)}, old: {np.mean(o_inter)}')

import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import f_oneway as anova

sb.boxplot([y_wb,m_wb,o_wb,y_intra,m_intra,o_intra,y_inter,m_inter,o_inter], palette=['blue', 'red','green','blue', 'red','green','blue', 'red','green'], width = 0.3)
plt.xticks([0,1, 2, 3, 4, 5, 6, 7, 8],['WB Y', 'WB M', 'WB O', 'Intra Y', 'Intra M','Intra O','Inter Y', 'Inter M','Inter O'])
plt.title(exp)
plt.show()

F, p = anova(y_wb,m_wb,o_wb)
print(f'WB: F = {F}, p = {p}')
F, p = anova(y_intra,m_intra,o_intra)
print(f'Intra: F = {F}, p = {p}')
F, p = anova(y_inter,m_inter,o_inter)
print(f'Inter: F = {F}, p = {p}')

