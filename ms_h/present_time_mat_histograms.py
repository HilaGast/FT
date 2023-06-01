import matplotlib.pyplot as plt
import numpy as np
import glob, os


main_fol = 'F:\Hila\TDI\siemens'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}')
exp = 'D60d11'
atlas = 'bnacor'
mat_type = 'time_th3'
control_group_vals = []
ms_group_vals = []
medians={}
for subj_fol in all_subj_fol:
    if 'group' in subj_fol or 'surfaces' in subj_fol:
        continue
    subj = subj_fol.split(os.sep)[-2]
    print(subj)
    if subj.startswith('C'):
        group = 'control'
    else:
        group = 'patient'
    mat = np.load(f'{subj_fol}{exp}{os.sep}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy')
    if group == 'control':
        control_group_vals.extend(mat[~np.isnan(mat)])
        #plt.hist(mat[~np.isnan(mat)], bins=50, histtype='step', color='blue',linewidth=2, range=(0,500))
        #plt.show()

    elif group == 'patient':
        ms_group_vals.extend(mat[~np.isnan(mat)])
        # plt.hist(mat[~np.isnan(mat)], bins=50, histtype='step', color='red',linewidth=2, range=(0,500))
        # plt.title(subj)
        # plt.show()
    medians[subj] = np.nanmedian(mat[mat>0])

plt.hist(control_group_vals, bins=50, histtype='step', color='blue',linewidth=2, density=True, range=(0,500))
plt.hist(ms_group_vals, bins=50, histtype='step', color='red',linewidth=2, density = True, range=(0,500))
plt.show()
