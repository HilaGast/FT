from Tractography.connectivity_matrices import *
import glob,os
subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
atlas = 'yeo7_200'
for sl in subj_list:
    if not os.path.exists(f'{sl}cm{os.sep}dist_{atlas}_cm_ord.npy'):
        diff_file = 'data.nii.gz'
        try:
            cm = WeightConMat(weight_by='dist', atlas=atlas, diff_file=diff_file, subj_folder=sl,
                              tract_name='HCP_tracts.tck')
            cm.save_cm(fig_name=f'dist_{atlas}', mat_type='cm_ord')
        except FileNotFoundError:
            try:
                diff_file = 'data.nii'
                cm = WeightConMat(weight_by='dist', atlas=atlas, diff_file=diff_file, subj_folder=sl,
                                  tract_name='HCP_tracts.tck')
                cm.save_cm(fig_name=f'dist_{atlas}', mat_type='cm_ord')
            except FileNotFoundError:
                continue
