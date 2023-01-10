import os, glob
import numpy as np
from network_analysis.global_network_properties import get_efficiency
from network_analysis.norm_cm_by_atlas_areas import norm_mat
from scipy.stats import skew
import matplotlib.pyplot as plt

weight_mat = 'add_bna_cm_ord.npy'
h_subj_list = glob.glob(f'F:\Hila\siemens\C*[0-9]_*[0-9]{os.sep}D31d18{os.sep}cm{os.sep}{weight_mat}')
ms_subj_list = glob.glob(f'F:\Hila\siemens\T*[0-9]_*[0-9]{os.sep}D31d18{os.sep}cm{os.sep}{weight_mat}')

h_eff=[]
ms_eff=[]

for sl in h_subj_list:

    cm = np.load(sl)
    h_eff.append(get_efficiency(cm))

for sl in ms_subj_list:

    cm = np.load(sl)
    ms_eff.append(get_efficiency(cm))



plt.hist(h_eff,15, color=[0.2, 0.7, 0.6], histtype='step',linewidth=3, range=(0, 0.02))
plt.hist(ms_eff,15, color=[0.3, 0.3, 0.5], histtype='step',linewidth=3, range=(0, 0.02))

#plt.legend(['Num Eglob','ADD Eglob','NumxADD Eglob'])
plt.legend([f'Healthy Eglob \n Mean: {np.round(np.nanmean(h_eff),3)}, STD: {np.round(np.nanstd(h_eff),3)} \n Skewness: {np.round(skew(h_eff, nan_policy="omit"),3)}',
            f'MS Eglob \n Mean: {np.round(np.nanmean(ms_eff),3)}, STD: {np.round(np.nanstd(ms_eff),3)} \n Skewness: {np.round(skew(ms_eff, nan_policy="omit"),3)}'])
plt.show()