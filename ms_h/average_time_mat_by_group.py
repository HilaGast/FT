import glob, os
from average_con_mat import calc_avg_mat
import matplotlib.pyplot as plt
import numpy as np

def average_time_mat(group_subj, group_name, main_fol, mat_type, atlas_type):
    calc_avg_mat(group_subj,mat_type,main_fol, calc_type='median', atlas_type = atlas_type, adds_for_file_name=group_name)

if __name__ == '__main__':
    main_fol = 'F:\Hila\TDI\siemens'
    exp = 'D60d11'
    atlas_type = 'bnacor'
    mat_type = 'time_th3'
    ms_subj_fol = glob.glob(f'{main_fol}{os.sep}T*{os.sep}{exp}')
    h_subj_fol = glob.glob(f'{main_fol}{os.sep}C*{os.sep}{exp}')
    average_time_mat(ms_subj_fol, f'{exp}_ms', main_fol, mat_type, atlas_type)
    average_time_mat(h_subj_fol, f'{exp}_h', main_fol, mat_type, atlas_type)

    a = np.load(rf'F:\Hila\TDI\siemens\median_{mat_type}_{atlas_type}_{exp}_h.npy')
    b = np.load(rf'F:\Hila\TDI\siemens\median_{mat_type}_{atlas_type}_{exp}_ms.npy')

    plt.hist(a[a > 0], bins=50, color='blue', alpha=0.2, range=(0, 500))
    plt.hist(b[b > 0], bins=50, color='red', alpha=0.2, range=(0, 500))
    plt.hist(a[a > 0], bins=50, histtype='step', color='blue', linewidth=2, range=(0, 500))
    plt.hist(b[b > 0], bins=50, histtype='step', color='red', linewidth=2, range=(0, 500))
    plt.legend(['Healthy', 'MS'])
    plt.show()

