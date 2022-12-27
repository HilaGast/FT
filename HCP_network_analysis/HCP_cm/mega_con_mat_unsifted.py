from Tractography.connectivity_matrices import *
import glob,os
from shutil import copy

shortlist = glob.glob(f'G:\data\V7\HCP\*{os.sep}')

for sl in shortlist:
    num = sl.split(os.sep)[-2]
    if not os.path.exists(rf'G:\data\V7\HCP\{num}\streamlines\HCP_tracts_unsifted.tck'):
        source_file = rf'Y:\qnap_hcp\{num}\tracts_unsifted\HCP.tck'
        dest_file = rf'G:\data\V7\HCP\{num}\streamlines\HCP_tracts_unsifted.tck'
        try:
            copy(source_file, dest_file)
        except FileNotFoundError:
            continue
    if not os.path.exists(f'{sl}cm{os.sep}fa_mega_unsifted_cm_ord.npy'):
        diff_file = 'data.nii.gz'
        try:

            cm = ConMat(atlas='mega', diff_file=diff_file,subj_folder=sl, tract_name='HCP_tracts_unsifted.tck')
            cm.save_cm(fig_name='num_mega_unsifted', mat_type='cm_ord')
            cm.draw_con_mat(mat_type='cm_ord',show=False)
        except FileNotFoundError:
            try:
                diff_file = 'data.nii'
                cm = ConMat(atlas='mega', diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts_unsifted.tck')
                cm.save_cm(fig_name='num_mega_unsifted', mat_type='cm_ord')
                cm.draw_con_mat(mat_type='cm_ord',show=False)
            except FileNotFoundError:
                 continue
        cm = WeightConMat(weight_by='data_FA', atlas='mega',diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts_unsifted.tck')
        cm.save_cm(fig_name='fa_mega_unsifted', mat_type='cm_ord')
        cm.draw_con_mat(mat_type='cm_ord',show=False)

    if not os.path.exists(f'{sl}cm{os.sep}add_mega_unsifted_cm_ord.npy'):
        cm = WeightConMat(weight_by='data_3_2_AxPasi7', atlas='mega',diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts_unsifted.tck')
        cm.save_cm(fig_name='add_mega_unsifted', mat_type='cm_ord')
        cm.draw_con_mat(mat_type='cm_ord',show=False)

