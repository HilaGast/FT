import os, glob
import numpy as np
from network_analysis.global_network_properties import get_efficiency
from draw_scatter_fit import draw_scatter_fit
shortlist = glob.glob(f'F:\data\V7\HCP\*{os.sep}')
add_eff=[]
num_eff=[]
for sl in shortlist:
    #subjnum = str.split(sl, os.sep)[2]
    #dir_name = f'F:\data\V7\HCP{os.sep}{subjnum}'
    dir_name = sl[:-1]


    add_cm = np.load(f'{dir_name}{os.sep}cm_add.npy')
    add_eff.append(get_efficiency(add_cm))

    num_cm = np.load(f'{dir_name}{os.sep}cm_num.npy')
    num_eff.append(get_efficiency(num_cm))

draw_scatter_fit(num_eff,add_eff, comp_reg=True, ttl='Eglob ADD vs Eglob Num')

