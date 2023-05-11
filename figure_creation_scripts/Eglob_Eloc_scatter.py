import os, glob
import numpy as np
from network_analysis.global_network_properties import get_efficiency
from network_analysis.nodes_network_properties import get_local_efficiency
from network_analysis.norm_cm_by_atlas_areas import norm_mat
from scipy.stats import skew
import matplotlib.pyplot as plt

from network_analysis.norm_cm_by_random_mat import make_n_rand_mat

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
add_eff=[]
num_eff=[]
fa_eff=[]

add_loceff=[]
num_loceff=[]
fa_loceff=[]

th = 'HistMatch'
atlas = 'yeo7_200'
for sl in subj_list[:100]:

    add_cm = np.load(f'{sl}cm{os.sep}{atlas}_ADD_{th}_SC_cm_ord.npy')
    add_eff.append(get_efficiency(add_cm))
    add_loceff.append(np.nanmean(get_local_efficiency(add_cm)))


    num_cm = np.load(f'{sl}cm{os.sep}{atlas}_Num_{th}_SC_cm_ord.npy')
    num_eff.append(get_efficiency(num_cm))
    num_loceff.append(np.nanmean(get_local_efficiency(num_cm)))

    fa_cm = np.load(f'{sl}cm{os.sep}{atlas}_FA_{th}_SC_cm_ord.npy')
    fa_eff.append(get_efficiency(fa_cm))
    fa_loceff.append(np.nanmean(get_local_efficiency(fa_cm)))

plt.scatter(num_loceff, num_eff, color=[0.2, 0.7, 0.6], s=10)
plt.scatter(add_loceff, add_eff, color=[0.8, 0.5, 0.3], s=10)
plt.scatter(fa_loceff, fa_eff, color=[0.3, 0.3, 0.5], s=10)

plt.legend(['Num', 'ADD', 'FA'])

plt.xlabel('Local Efficiency')
plt.ylabel('Global Efficiency')
plt.show()
