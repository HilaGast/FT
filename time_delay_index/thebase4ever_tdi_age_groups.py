import pandas as pd
import os, glob
import numpy as np

from time_delay_index.average_hcp_tdi import calc_avg_mat

table_years = pd.read_excel(r'F:\Hila\TDI\TheBase4Ever subjects.xlsx')
atlas_type = 'yeo7_100'
mat_type = 'time_th3'
subj_y = []
subj_o = []
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
    mat = np.load(f'{subj}cm{os.sep}{atlas_type}_{mat_type}_cm_ord.npy')
    mat[mat==0] = np.nan
    if age < 40:
        subj_y.append(subj)
    elif age > 50:
        subj_o.append(subj)

calc_avg_mat(subj_y, mat_type,main_fol+'\cm', 'median', atlas_type, adds_for_file_save='_young')
calc_avg_mat(subj_o, mat_type,main_fol+'\cm', 'median', atlas_type, adds_for_file_save='_old')
