import os, glob
import numpy as np
from network_analysis.global_network_properties import get_efficiency
from draw_scatter_fit import draw_scatter_fit
subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
add_eff=[]
num_eff=[]
fa_eff=[]
add_num_eff=[]
th = '_histmatch_th'

for sl in subj_list:

    add_cm = np.load(f'{sl}cm{os.sep}add_bna_cm_ord{th}.npy')

    num_cm = np.load(f'{sl}cm{os.sep}num_bna_cm_ord{th}.npy')

    fa_cm = np.load(f'{sl}cm{os.sep}fa_bna_cm_ord{th}.npy')

    add_eff.append(get_efficiency(add_cm))
    num_eff.append(get_efficiency(num_cm))
    fa_eff.append(get_efficiency(fa_cm))
    #add_num_mat = add_cm * num_cm
    #add_num_eff.append(get_efficiency(add_num_mat))

draw_scatter_fit(num_eff,add_eff, comp_reg=True, ttl='Eglob ADD vs Eglob Num', remove_outliers= False)
draw_scatter_fit(num_eff,fa_eff, comp_reg=True, ttl='Eglob FA vs Eglob Num', remove_outliers= False)
draw_scatter_fit(fa_eff,add_eff, comp_reg=True, ttl='Eglob ADD vs Eglob FA', remove_outliers= False)

