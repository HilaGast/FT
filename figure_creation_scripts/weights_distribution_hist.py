import os, glob
import numpy as np
from network_analysis.global_network_properties import get_efficiency
from network_analysis.norm_cm_by_atlas_areas import norm_mat
from scipy.stats import skew
import matplotlib.pyplot as plt


th = 'HistmatchNorm'
atlas = 'yeo7_200'


add = np.load(f'G:\data\V7\HCP\cm{os.sep}average_{atlas}_ADD_{th}_SC.npy').reshape(-1)

num = np.load(f'G:\data\V7\HCP\cm{os.sep}average_{atlas}_Num_{th}_SC.npy').reshape(-1)

fa = np.load(f'G:\data\V7\HCP\cm{os.sep}average_{atlas}_FA_{th}_SC.npy').reshape(-1)

#dist = np.load(f'G:\data\V7\HCP\cm{os.sep}average_{atlas}_Dist_{th}_SC.npy').reshape(-1)

add[add==0] = np.nan
num[num==0] = np.nan
fa[fa==0] = np.nan
#dist[dist==0] = np.nan

plt.hist(num,30, color=[0.2, 0.7, 0.6], histtype='step',linewidth=3, range=(0, 1))
plt.hist(add,30, color=[0.8, 0.5, 0.3], histtype='step',linewidth=3, range=(0, 1))
plt.hist(fa,30, color=[0.3, 0.3, 0.5], histtype='step',linewidth=3, range=(0, 1))
#plt.hist(dist,15, color=[0.2, 0.5, 0.8], histtype='step',linewidth=3, range=(0, 100))

#plt.legend(['Num Eglob','ADD Eglob','NumxADD Eglob'])
# plt.legend([f'Num \n Mean: {np.round(np.nanmean(num),3)}, STD: {np.round(np.nanstd(num),3)} \n Skewness: {np.round(skew(num, nan_policy="omit"),3)}',
#             f'ADD \n Mean: {np.round(np.nanmean(add),3)}, STD: {np.round(np.nanstd(add),3)} \n Skewness: {np.round(skew(add, nan_policy="omit"),3)}',
#             f'FA \n Mean: {np.round(np.nanmean(fa), 3)}, STD: {np.round(np.nanstd(fa), 3)} \n Skewness: {np.round(skew(fa, nan_policy="omit"), 3)}',
#             f'Dist \n Mean: {np.round(np.nanmean(dist),3)}, STD: {np.round(np.nanstd(dist),3)} \n Skewness: {np.round(skew(dist, nan_policy="omit"),3)}'])
plt.legend([f'Num', f'ADD', f'FA'])

plt.show()