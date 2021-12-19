import os, glob
from network_analysis.global_network_properties import *
from parcellation.nodes_add_correlation_to_age import age_var
import numpy as np


main_subj_folders = 'F:\data\V7\TheBase4Ever'

eff_num = []
#eff_fa = []
eff_add = []
subj_idx = []
eff_add_num = []

for sub in glob.glob(f'{main_subj_folders}{os.sep}*{os.sep}'):

    num_mat_name = sub + 'non-weighted_wholebrain_5d_labmask_bna_nonnorm.npy'
    if os.path.exists(num_mat_name):
        subj_idx.append(True)
        num_mat = np.load(num_mat_name)
        eff_num.append(get_efficiency(cm=num_mat))

        #fa_mat_name = sub + 'weighted_wholebrain_5d_labmask_yeo7_200_FA_nonnorm.npy'
        #fa_mat = np.load(fa_mat_name)
        #eff_fa.append(get_efficiency(cm=fa_mat))

        add_mat_name = sub + 'weighted_wholebrain_5d_labmask_bna_nonnorm.npy'
        add_mat = np.load(add_mat_name)
        eff_add.append(get_efficiency(cm=add_mat))

        add_num_mat = add_mat*num_mat
        eff_add_num.append(get_efficiency(add_num_mat))



    else:
        subj_idx.append(False)

ages = age_var(main_subj_folders, subj_idx)


from draw_scatter_fit import draw_scatter_fit as dsf

dsf(ages, eff_num, deg=2, norm_x=False)
dsf(ages, eff_add, deg=2, norm_x=False)

dsf(ages, eff_num, norm_x=False, comp_reg=True)
dsf(ages, eff_add, norm_x=False, comp_reg=True)
