import pandas as pd
import os, glob
import numpy as np

from HCP_network_analysis.group_avarage_mat import calc_group_average_mat
from calc_corr_statistics.pearson_r_calc import calc_corr
import matplotlib.pyplot as plt

from ms_h.average_time_mat_by_group import average_time_mat
from ms_h.present_time_mat_by_hemisphere import divide_mat_to_inter_intra_hemi_mats

main_fol = 'G:\data\V7\HCP'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}*[0-9]{os.sep}')
atlas_type = 'yeo7_200'
mat_type = 'time_th3'

ages = []
tdi = []
tdi_inter = []
tdi_intra = []
table1 = pd.read_csv('G:\data\V7\HCP\HCP_demographic_data.csv')

subj_y = []
subj_o = []
all_subj_cm = glob.glob(f'{main_fol}{os.sep}*[0-9]{os.sep}cm{os.sep}{atlas_type}_{mat_type}_Org_SC_cm_ord.npy')
mean_mat = calc_group_average_mat(all_subj_cm, atlas_type, type='median')
np.save(f'{main_fol}{os.sep}cm{os.sep}median_{atlas_type}_{mat_type}_Org_SC.npy', mean_mat)

for subj in all_subj_fol:
    subj_number = subj.split(os.sep)[-2]
    age = int(table1['Age_in_Yrs'][table1['Subject']==int(subj_number)].values)
    ages.append(age)

    mat = np.load(f'{subj}cm{os.sep}{atlas_type}_{mat_type}_Org_SC_cm_ord.npy')
    mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat, atlas_type)
    mat[mat == 0] = np.nan
    tdi.append(np.nanmedian(mat))
    mat_inter[mat_inter == 0] = np.nan
    tdi_inter.append(np.nanmedian(mat_inter))
    mat_intra[mat_intra == 0] = np.nan
    tdi_intra.append(np.nanmedian(mat_intra))
    if age < 24:
        subj_y.append(f'{subj}{os.sep}cm{os.sep}{atlas_type}_{mat_type}_Org_SC_cm_ord.npy')
    elif age>35:
        subj_o.append(f'{subj}{os.sep}cm{os.sep}{atlas_type}_{mat_type}_Org_SC_cm_ord.npy')

mean_mat = calc_group_average_mat(subj_y, atlas_type, type='median')
np.save(f'{main_fol}{os.sep}cm{os.sep}median_{atlas_type}_{mat_type}_Org_SC_young(22-23).npy', mean_mat)
mean_mat = calc_group_average_mat(subj_o, atlas_type, type='median')
np.save(f'{main_fol}{os.sep}cm{os.sep}median_{atlas_type}_{mat_type}_Org_SC_old(36-37).npy', mean_mat)

a = np.load(rf'{main_fol}{os.sep}cm{os.sep}median_{atlas_type}_{mat_type}_Org_SC_young(22-23).npy')
c = np.load(rf'{main_fol}{os.sep}cm{os.sep}median_{atlas_type}_{mat_type}_Org_SC_old(36-37).npy')
plt.hist(a[a > 0], bins=50, color='blue', alpha=0.2, range=(0, 0.15), density=True)
plt.hist(c[c > 0], bins=50, color='green', alpha=0.2, range=(0, 0.15), density=True)
plt.hist(a[a > 0], bins=50, histtype='step', color='blue', linewidth=2, range=(0, 0.15), density=True)
plt.hist(c[c > 0], bins=50, histtype='step', color='green', linewidth=2, range=(0, 0.15), density=True)
plt.legend(['Younger (22-29)',  'Older (30-37)'])
plt.show()

r, p = calc_corr(ages, tdi)
print(f'WB: \n r:{r} - p:{p}')
plt.plot(ages, tdi, 'o')
plt.title('WB')
plt.show()

r, p = calc_corr(ages, tdi_inter)
print(f'Inter: \n r:{r} - p:{p}')
plt.plot(ages, tdi_inter, 'o')
plt.title('Inter')
plt.show()

r, p = calc_corr(ages, tdi_intra)
print(f'Intra: \n r:{r} - p:{p}')
plt.plot(ages, tdi_intra, 'o')
plt.title('Intra')
plt.show()





