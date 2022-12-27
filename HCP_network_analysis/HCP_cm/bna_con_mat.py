from Tractography.connectivity_matrices import *
import glob,os

shortlist = glob.glob(f'G:\data\V7\HCP\*{os.sep}')

for sl in shortlist:
    if not os.path.exists(f'{sl}cm{os.sep}num_bna_cm_ord.npy'):

        diff_file = 'data.nii.gz'
        try:
            cm = ConMat(atlas='bna', diff_file=diff_file,subj_folder=sl, tract_name='HCP_tracts.tck')
            cm.save_cm(fig_name='num_bna', mat_type='cm_ord')
            cm.draw_con_mat(mat_type='cm_ord', show=False)
        except FileNotFoundError:
            try:
                diff_file = 'data.nii'
                cm = ConMat(atlas='bna', diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts.tck')
                cm.save_cm(fig_name='num_bna', mat_type='cm_ord')
                cm.draw_con_mat(mat_type='cm_ord', show=False)
            except FileNotFoundError:
                continue
    if not os.path.exists(f'{sl}cm{os.sep}fa_bna_cm_ord.npy'):
        cm = WeightConMat(weight_by='data_FA', atlas='bna',diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts.tck')
        cm.save_cm(fig_name='fa_bna', mat_type='cm_ord')
        cm.draw_con_mat(mat_type='cm_ord',show=False)

    if not os.path.exists(f'{sl}cm{os.sep}add_bna_cm_ord.npy'):
        cm = WeightConMat(weight_by='data_3_2_AxPasi7', atlas='bna',diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts.tck')
        cm.save_cm(fig_name='add_bna', mat_type='cm_ord')
        cm.draw_con_mat(mat_type='cm_ord',show=False)


