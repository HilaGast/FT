import os,glob
import numpy as np
from basic_statistics import ttest
from calc_corr_statistics.pearson_r_calc import multi_comp_correction

main_fol = 'F:\Hila\TDI\siemens'
exp = 'D60d11'
atlas = 'bnacor'
mat_type = 'time_th3'
h_mat_files = glob.glob(f'{main_fol}{os.sep}C*{os.sep}{exp}{os.sep}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy')
ms_mat_files = glob.glob(f'{main_fol}{os.sep}T*{os.sep}{exp}{os.sep}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy')
idx = np.load(rf'F:\Hila\TDI\siemens\group_cm\{atlas}_cm_ord_lookup.npy')
H = np.zeros((len(idx),len(idx),len(h_mat_files)))
MS = np.zeros((len(idx),len(idx),len(ms_mat_files)))

for h_mat_file in h_mat_files:
    mat = np.load(h_mat_file)
    H[:,:,h_mat_files.index(h_mat_file)] = mat

for ms_mat_file in ms_mat_files:
    mat = np.load(ms_mat_file)
    MS[:,:,ms_mat_files.index(ms_mat_file)] = mat
H[np.isnan(H)] = 0
MS[np.isnan(MS)] = 0
t,p = ttest(H,MS,type = 'independent',alternative='two-sided')

np.save(rf'F:\Hila\TDI\siemens\group_cm\{exp}_{mat_type}_{atlas}_cm_ord_ttest',t)
np.save(rf'F:\Hila\TDI\siemens\group_cm\{exp}_{mat_type}_{atlas}_cm_ord_pval',p)


t,p,t_th = multi_comp_correction(t,p,0.01)
t_th[t_th==0] = np.nan
np.save(rf'F:\Hila\TDI\siemens\group_cm\{exp}_{mat_type}_{atlas}_cm_ord_ttest_fdr_poin01',t_th)
