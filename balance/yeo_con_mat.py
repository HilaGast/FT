from Tractography.connectivity_matrices import *
import glob,os

shortlist = glob.glob(rf'F:\Hila\balance\e*\before{os.sep}*{os.sep}')
atlas = 'yeo7_200'
for sl in shortlist:
    tract_name = glob.glob(f'{sl}streamlines{os.sep}*_wholebrain_4d_labmask.trk')[0].split(os.sep)[-1]
    if not os.path.exists(f'{sl}cm{os.sep}num_{atlas}_cm_ord.npy'):

        diff_file = 'diff_corrected.nii'
        try:
            cm = ConMat(atlas=atlas, diff_file=diff_file,subj_folder=sl, tract_name=tract_name)
            cm.save_cm(fig_name=f'num_{atlas}', mat_type='cm_ord')

        except FileNotFoundError:
            continue
    if not os.path.exists(f'{sl}cm{os.sep}fa_{atlas}_cm_ord.npy'):
        cm = WeightConMat(weight_by='diff_corrected_FA', atlas=atlas,diff_file=diff_file, subj_folder=sl, tract_name=tract_name)
        cm.save_cm(fig_name=f'fa_{atlas}', mat_type='cm_ord')

    if not os.path.exists(f'{sl}cm{os.sep}add_{atlas}_cm_ord.npy'):
        cm = WeightConMat(weight_by='diff_corrected_3_2_AxPasi7', atlas=atlas,diff_file=diff_file, subj_folder=sl, tract_name=tract_name)
        cm.save_cm(fig_name=f'add_{atlas}', mat_type='cm_ord')


