import os, glob
import numpy as np
from network_analysis.global_network_properties import get_rich_club_curve

main_subj_folders = 'F:\data\V7\TheBase4Ever'

add_rc=[]
num_rc=[]

for sub in glob.glob(f'{main_subj_folders}{os.sep}*{os.sep}'):


    add_cm = np.load(f'{sub}weighted_wholebrain_5d_labmask_bna_nonnorm.npy')
    add_rc.append(get_rich_club_curve(add_cm))

    num_cm = np.load(f'{sub}non-weighted_wholebrain_5d_labmask_bna_nonnorm.npy')
    num_rc.append(get_rich_club_curve(num_cm))


num_rc = np.asarray(num_rc)
num_rc = num_rc[:,1:]
add_rc = np.asarray(add_rc)
add_rc = add_rc[:,1:]

num_rc_mean = np.nanmean(num_rc, axis=0)
add_rc_mean = np.nanmean(add_rc, axis=0)
k = list(range(1,20))

import matplotlib.pyplot as plt

plt.plot(k,num_rc_mean,'r')
plt.plot(k, add_rc_mean,'b')
plt.legend(['num','add'])
plt.show()
