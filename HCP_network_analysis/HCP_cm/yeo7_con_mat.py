from Tractography.connectivity_matrices import *
import glob,os

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
atlas = 'yeo7_200'
for sl in subj_list:
    if not os.path.exists(f'{sl}cm{os.sep}num_{atlas}_cm_ord.npy'):

        diff_file = 'data.nii.gz'
        try:
            cm = ConMat(atlas=atlas, diff_file=diff_file,subj_folder=sl, tract_name='HCP_tracts.tck')
            cm.save_cm(fig_name=f'num_{atlas}', mat_type='cm_ord')
            cm.draw_con_mat(mat_type='cm_ord', show=False)
        except FileNotFoundError:
            try:
                diff_file = 'data.nii'
                cm = ConMat(atlas=atlas, diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts.tck')
                cm.save_cm(fig_name=f'num_{atlas}', mat_type='cm_ord')
                cm.draw_con_mat(mat_type='cm_ord', show=False)
            except FileNotFoundError:
                continue
    if not os.path.exists(f'{sl}cm{os.sep}fa_{atlas}_cm_ord.npy'):
        cm = WeightConMat(weight_by='data_FA', atlas=atlas,diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts.tck')
        cm.save_cm(fig_name=f'fa_{atlas}', mat_type='cm_ord')
        cm.draw_con_mat(mat_type='cm_ord',show=False)

    if not os.path.exists(f'{sl}cm{os.sep}add_{atlas}_cm_ord.npy'):
        cm = WeightConMat(weight_by='data_3_2_AxPasi7', atlas=atlas,diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts.tck')
        cm.save_cm(fig_name=f'add_{atlas}', mat_type='cm_ord')
        cm.draw_con_mat(mat_type='cm_ord',show=False)


