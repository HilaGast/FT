import os, glob
import numpy as np
from network_analysis.global_network_properties import get_efficiency
from network_analysis.norm_cm_by_atlas_areas import norm_mat
import matplotlib.pyplot as plt

subj_list = glob.glob(f'F:\data\V7\HCP\*{os.sep}')
add_eff=[]
num_eff=[]
add_num_eff=[]
for sl in subj_list:

    add_cm = np.load(f'{sl}cm{os.sep}add_bna_cm_ord.npy')
    add_eff.append(get_efficiency(add_cm))

    num_cm = np.load(f'{sl}cm{os.sep}num_bna_cm_ord_corrected.npy')
    #num_cm = norm_mat(sl,num_cm,'bna')
    num_eff.append(get_efficiency(num_cm))

    add_num_mat = add_cm * num_cm
    add_num_eff.append(get_efficiency(add_num_mat))

plt.hist(num_eff,15, color=[0.2, 0.7, 0.6], histtype='step',linewidth=4)
plt.hist(add_eff,15, color=[0.2, 0.5, 0.8], histtype='step',linewidth=4)
#plt.hist(add_num_eff,15, color=[0.8, 0.5, 0.3], histtype='step',linewidth=4)
#plt.legend(['Num Eglob','ADD Eglob','NumxADD Eglob'])
plt.legend(['Num Eglob','ADD Eglob'])
plt.show()