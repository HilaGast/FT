from basic_statistics import ttest
from ms_h.present_time_mat_by_hemisphere import divide_mat_to_inter_intra_hemi_mats
import numpy as np

vals_inter=[]
vals_intra=[]
mat_num = np.load(r'G:\data\V7\HCP\cm\median_yeo7_100_Num_Org_SC.npy')
mat_dist = np.load(r'G:\data\V7\HCP\cm\median_yeo7_100_Dist_Org_SC.npy')
mat_add = np.load(r'G:\data\V7\HCP\cm\median_yeo7_100_ADD_Org_SC.npy')
mat_tdi = np.load(r'G:\data\V7\HCP\cm\median_yeo7_100_time_th3_Org_SC.npy')

mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat_num, 'yeo7_100')
mat_intra[mat_intra == 0] = np.nan
mat_inter[mat_inter == 0] = np.nan
t, p = ttest(mat_intra[~np.isnan(mat_intra)], mat_inter[~np.isnan(mat_inter)], type='independent',
             alternative='two-sided')
print(f'Num: t = {t}, p = {p}')
vals_inter.append(mat_inter[~np.isnan(mat_inter)]/np.nanmax(mat_num))
vals_intra.append(mat_intra[~np.isnan(mat_intra)]/np.nanmax(mat_num))

mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat_dist, 'yeo7_100')
mat_intra[mat_intra == 0] = np.nan
mat_inter[mat_inter == 0] = np.nan
t, p = ttest(mat_intra[~np.isnan(mat_intra)], mat_inter[~np.isnan(mat_inter)], type='independent',
                alternative='two-sided')
print(f'Dist: t = {t}, p = {p}')

vals_inter.append(mat_inter[~np.isnan(mat_inter)]/np.nanmax(mat_dist))
vals_intra.append(mat_intra[~np.isnan(mat_intra)]/np.nanmax(mat_dist))

mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat_add, 'yeo7_100')
mat_intra[mat_intra == 0] = np.nan
mat_inter[mat_inter == 0] = np.nan
t, p = ttest(mat_intra[~np.isnan(mat_intra)], mat_inter[~np.isnan(mat_inter)], type='independent',
                alternative='two-sided')
print(f'ADD: t = {t}, p = {p}')

vals_inter.append(mat_inter[~np.isnan(mat_inter)]/np.nanmax(mat_add))
vals_intra.append(mat_intra[~np.isnan(mat_intra)]/np.nanmax(mat_add))

mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat_tdi, 'yeo7_100')
mat_intra[mat_intra == 0] = np.nan
mat_inter[mat_inter == 0] = np.nan
t, p = ttest(mat_intra[~np.isnan(mat_intra)], mat_inter[~np.isnan(mat_inter)], type='independent',
                alternative='two-sided')
print(f'TDI: t = {t}, p = {p}')

vals_inter.append(mat_inter[~np.isnan(mat_inter)]/np.nanmax(mat_tdi))
vals_intra.append(mat_intra[~np.isnan(mat_intra)]/np.nanmax(mat_tdi))

import matplotlib.pyplot as plt
plt.figure(figsize=(6,3.5))
plt.bar([1,2,3,4,5,6,7,8], [np.nanmean(vals_intra[0]), np.nanmean(vals_inter[0]), np.nanmean(vals_intra[1]), np.nanmean(vals_inter[1]), np.nanmean(vals_intra[2]), np.nanmean(vals_inter[2]), np.nanmean(vals_intra[3]), np.nanmean(vals_inter[3])],yerr=[np.nanstd(vals_intra[0]), np.nanstd(vals_inter[0]), np.nanstd(vals_intra[1]), np.nanstd(vals_inter[1]), np.nanstd(vals_intra[2]), np.nanstd(vals_inter[2]), np.nanstd(vals_intra[3]), np.nanstd(vals_inter[3])],color=['b','r','b','r','b','r','b','r'],width=0.5)
plt.xticks([1,2,3,4,5,6,7,8], ['Num\nIntra', 'Num\nInter', 'Dist\nIntra', 'Dist\nInter', 'ADD\nIntra', 'ADD\nInter', 'TDI\nIntra', 'TDI\nInter'])
plt.box(False)
plt.show()