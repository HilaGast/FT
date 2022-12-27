from fsl.file_prep import os_path_2_fsl, create_inv_mat, flirt_primary_guess, fnirt_from_atlas_2_subj, apply_fnirt_warp_on_label
import glob, os

main_fol = 'F:\Hila\siemens'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}')

experiments = ['D31d18','D45d13','D60d11']
atlas_label = r'G:\data\atlases\BNA\BN_Atlas_274_combined_1mm.nii'
atlas_template = r'G:\data\atlases\BNA\MNI152_T1_1mm.nii'
atlas_label = os_path_2_fsl(atlas_label)
atlas_template = os_path_2_fsl(atlas_template)

for fol in all_subj_fol:
    for experiment in experiments:
        exp_fol = os_path_2_fsl(f'{fol}{experiment}{os.sep}')
        subj_mprage = os_path_2_fsl(f'{fol}MPRAGE.nii')
        diff_file_name = f'{exp_fol}diff_corrected_{experiment}.nii'



        in_brain = subj_mprage[:-4]
        out_brain = subj_mprage[:-4]+'_brain'
        diff_file_1st = diff_file_name[:-4]+'_1st'


        ''' Registration from MPRAGE to 1st CHARMED scan using inverse matrix of CHARMED to MPRAGE registration:
        From CHARMED to MPRAGE:'''
        subj_first_charmed = diff_file_name[:-4] + '_1st.nii'
        out_registered = diff_file_name[:-4] + '_1st_reg.nii'
        out_registered_mat = out_registered[:-4] + '.mat'

        '''Creation of inverse matrix:  '''
        inv_mat = create_inv_mat(out_registered_mat)

        '''From MPRAGE to CHARMED using the inverse matrix: '''
        out_registered = f'{exp_fol}mprage_reg.nii'

        '''Registration from atlas to regisered MPRAGE:
        flirt for atlas to registered MPRAGE for primary guess:  '''
        atlas_brain, atlas_registered_flirt, atlas_registered_flirt_mat = flirt_primary_guess(exp_fol, atlas_template,
                                                                                          out_registered)

        '''fnirt for atlas based on flirt results:    '''
        warp_name = fnirt_from_atlas_2_subj(exp_fol, out_registered, atlas_brain, atlas_registered_flirt_mat,
                                        cortex_only=False)

        '''apply fnirt warp on atlas labels:   '''
        atlas_labels_registered = apply_fnirt_warp_on_label(exp_fol, atlas_label, out_registered, warp_name)
