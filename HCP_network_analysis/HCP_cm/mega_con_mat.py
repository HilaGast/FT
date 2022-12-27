from Tractography.connectivity_matrices import *
import glob,os

shortlist = glob.glob(f'G:\data\V7\HCP\*{os.sep}')

for sl in shortlist:
    if not os.path.exists(f'{sl}cm{os.sep}num_mega_cm_ord.npy'):

        diff_file = 'data.nii.gz'
        try:
            cm = ConMat(atlas='mega', diff_file=diff_file,subj_folder=sl, tract_name='HCP_tracts.tck')
            cm.save_cm(fig_name='num_mega', mat_type='cm_ord')
            cm.draw_con_mat(mat_type='cm_ord', show=False)
        except FileNotFoundError:
            try:
                diff_file = 'data.nii'
                cm = ConMat(atlas='mega', diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts.tck')
                cm.save_cm(fig_name='num_mega', mat_type='cm_ord')
                cm.draw_con_mat(mat_type='cm_ord', show=False)
            except FileNotFoundError:
                continue
    if not os.path.exists(f'{sl}cm{os.sep}fa_mega_cm_ord.npy'):
        cm = WeightConMat(weight_by='data_FA', atlas='mega',diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts.tck')
        cm.save_cm(fig_name='fa_mega', mat_type='cm_ord')
        cm.draw_con_mat(mat_type='cm_ord',show=False)

    if not os.path.exists(f'{sl}cm{os.sep}add_mega_cm_ord.npy'):
        cm = WeightConMat(weight_by='data_3_2_AxPasi7', atlas='mega',diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts.tck')
        cm.save_cm(fig_name='add_mega', mat_type='cm_ord')
        cm.draw_con_mat(mat_type='cm_ord',show=False)


