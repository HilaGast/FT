import pandas as pd
import os, glob
import numpy as np
from calc_corr_statistics.pearson_r_calc import calc_corr
import matplotlib.pyplot as plt

from ms_h.present_time_mat_by_hemisphere import divide_mat_to_inter_intra_hemi_mats

table_years = pd.read_excel(r'F:\Hila\TDI\TheBase4Ever subjects.xlsx')
atlas_type = 'yeo7_100'
mat_type = 'Dist'
ages = []
tdi = []
tdi_intra = []
tdi_inter = []
main_fol = 'F:\Hila\TDI\TheBase4Ever'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}[0-9]*{os.sep}')

for subj in all_subj_fol:
    subj_name = subj.split(os.sep)[-2]
    subj_index = table_years['Scan File 1'].str.contains(subj_name)
    try:
        age = table_years['Age'][subj_index].values[0]
    except IndexError:
        subj_index = table_years['Scan File 2'].str.contains(subj_name)
        subj_index = subj_index[subj_index==True].index[0]
        age = table_years['Age'][subj_index]
    if 20<age<45:
        ages.append(age)
        mat = np.load(f'{subj}cm{os.sep}{atlas_type}_{mat_type}_cm_ord.npy')
        mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat, atlas_type)
        mat[mat==0] = np.nan
        mat_intra[mat_intra==0] = np.nan
        mat_inter[mat_inter==0] = np.nan
        tdi.append(np.nanmedian(mat))
        tdi_intra.append(np.nanmedian(mat_intra))
        tdi_inter.append(np.nanmedian(mat_inter))
    else:
        continue
ages.pop(-1)
tdi.pop(-1)
tdi_intra.pop(-1)
tdi_inter.pop(-1)

r, p = calc_corr(ages, tdi)
print(f'WB: \n r:{r} - p:{p}')
plt.plot(ages, tdi, 'o')
plt.title('WB')
plt.show()

r, p = calc_corr(ages, tdi_intra)
print(f'Intra: \n r:{r} - p:{p}')
plt.plot(ages, tdi_intra, 'o')
plt.title('Intra')
plt.show()

r, p = calc_corr(ages, tdi_inter)
print(f'Inter: \n r:{r} - p:{p}')
plt.plot(ages, tdi_inter, 'o')
plt.title('Inter')
plt.show()
