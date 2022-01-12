from Tractography.connectivity_matrices import *
import glob,os

shortlist = glob.glob(f'F:\data\V7\HCP\*{os.sep}')

for sl in shortlist:
    cm = ConMat(atlas='bna',subj_folder=sl, tract_name='HCP_tracts_unsifted.tck')
    cm.save_cm(fig_name='num_bna',mat_type='cm_ord')
    cm.draw_con_mat(mat_type='cm_ord')

    cm = WeightConMat(weight_by='3_2_AxPasi7',atlas='bna',subj_folder=sl, tract_name='HCP_tracts_unsifted.tck')
    cm.save_cm(fig_name='add_bna',mat_type='cm_ord')
    cm.draw_con_mat(mat_type='cm_ord')
