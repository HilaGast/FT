from Tractography.connectivity_matrices import *
import glob,os

shortlist = glob.glob(f'F:\data\V7\HCP\*{os.sep}')

for sl in shortlist:
    if not os.path.exists(f'{sl}cm{os.sep}add_bna_cm_ord.npy'):
        try:
            cm = ConMat(atlas='bna', diff_file='data.nii.gz',subj_folder=sl, tract_name='HCP_tracts.tck')
        except FileNotFoundError:
            continue
        cm.save_cm(fig_name='num_bna', mat_type='cm_ord')
        cm.draw_con_mat(mat_type='cm_ord')

        cm = WeightConMat(weight_by='data_3_2_AxPasi7', atlas='bna',diff_file='data.nii.gz', subj_folder=sl, tract_name='HCP_tracts.tck')
        cm.save_cm(fig_name='add_bna', mat_type='cm_ord')
        cm.draw_con_mat(mat_type='cm_ord',show=False)
