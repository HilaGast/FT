import os, glob
import numpy as np
from network_analysis.global_network_properties import get_efficiency
from network_analysis.norm_cm_by_atlas_areas import norm_mat
from scipy.stats import skew
import matplotlib.pyplot as plt

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
add_eff=[]
num_eff=[]
fa_eff=[]
dist_eff=[]
th = 'HistMatch'
atlas = 'yeo7_200'
for sl in subj_list:

    add_cm = np.load(f'{sl}cm{os.sep}{atlas}_ADD_{th}_SC_cm_ord.npy')
    add_eff.append(get_efficiency(add_cm))

    num_cm = np.load(f'{sl}cm{os.sep}{atlas}_Num_{th}_SC_cm_ord.npy')
    num_eff.append(get_efficiency(num_cm))

    fa_cm = np.load(f'{sl}cm{os.sep}{atlas}_FA_{th}_SC_cm_ord.npy')
    fa_eff.append(get_efficiency(fa_cm))

    dist_mat = np.load(f'{sl}cm{os.sep}{atlas}_Dist_{th}_SC_cm_ord.npy')
    dist_eff.append(get_efficiency(dist_mat))

plt.hist(num_eff,15, color=[0.2, 0.7, 0.6], histtype='step',linewidth=3, range=(0, 0.05))
plt.hist(add_eff,15, color=[0.2, 0.5, 0.8], histtype='step',linewidth=3, range=(0, 0.05))
plt.hist(fa_eff,15, color=[0.3, 0.3, 0.5], histtype='step',linewidth=3, range=(0, 0.05))
plt.hist(dist_eff,15, color=[0.8, 0.5, 0.3], histtype='step',linewidth=3, range=(0, 0.05))

#plt.legend(['Num Eglob','ADD Eglob','NumxADD Eglob'])
plt.legend([f'Num Eglob \n Mean: {np.round(np.nanmean(num_eff),3)}, STD: {np.round(np.nanstd(num_eff),3)} \n Skewness: {np.round(skew(num_eff, nan_policy="omit"),3)}',
            f'ADD Eglob \n Mean: {np.round(np.nanmean(add_eff),3)}, STD: {np.round(np.nanstd(add_eff),3)} \n Skewness: {np.round(skew(add_eff, nan_policy="omit"),3)}',
            f'FA Eglob \n Mean: {np.round(np.nanmean(fa_eff), 3)}, STD: {np.round(np.nanstd(fa_eff), 3)} \n Skewness: {np.round(skew(fa_eff, nan_policy="omit"), 3)}',
            f'Dist Eglob \n Mean: {np.round(np.nanmean(dist_eff),3)}, STD: {np.round(np.nanstd(dist_eff),3)} \n Skewness: {np.round(skew(dist_eff, nan_policy="omit"),3)}'])
plt.show()