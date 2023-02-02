import matplotlib.pyplot as plt
import numpy as np
import glob, os


main_fol = 'F:\Hila\siemens'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}')
exp = 'D31d18'
control_group_vals = []
ms_group_vals = []

for subj_fol in all_subj_fol:
    subj = subj_fol.split(os.sep)[-2]
    print(subj)
    if subj.startswith('C'):
        group = 'control'
    else:
        group = 'patient'
    mat = np.load(f'{subj_fol}{exp}{os.sep}cm{os.sep}time_th30_bnacor_cm_ord.npy')
    if group == 'control':
        control_group_vals.extend(mat[~np.isnan(mat)])
        #plt.hist(mat[~np.isnan(mat)], bins=50, histtype='step', color='blue',linewidth=2, range=(0,0.2))
        #plt.show()

    elif group == 'patient':
        ms_group_vals.extend(mat[~np.isnan(mat)])
        #plt.hist(mat[~np.isnan(mat)], bins=50, histtype='step', color='red',linewidth=2, range=(0,0.2))
        #plt.show()
plt.hist(control_group_vals, bins=50, histtype='step', color='blue',linewidth=2, range=(0,0.2), density=True)
plt.hist(ms_group_vals, bins=50, histtype='step', color='red',linewidth=2, range=(0,0.2), density = True)
plt.show()
