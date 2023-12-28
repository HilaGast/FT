from fsl.file_prep import fast_seg,os_path_2_fsl, create_inv_mat, reg_from_mprage_2_chm_inv
import glob, os

main_fol = 'F:\Hila\siemens\more'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}')

experiments = ['D31d18','D45d13','D60d11']

for fol in all_subj_fol:
    for experiment in experiments:
        exp_fol = os_path_2_fsl(f'{fol}{experiment}{os.sep}')
        subj_mprage = os_path_2_fsl(f'{fol}MPRAGE.nii')
        diff_file_name = f'{exp_fol}diff_corrected_{experiment}.nii'

        in_brain = subj_mprage[:-4]
        out_brain = subj_mprage[:-4]+'_brain'
        diff_file_1st = diff_file_name[:-4]+'_1st'

        # BET for MPRAGE:
        cmd = f'bash -lc "bet {in_brain} {out_brain} -f 0.45 -g -0.3"'
        cmd = cmd.replace(os.sep, '/')
        os.system(cmd)
        # save first corrected diff:
        cmd = fr'bash -lc "fslroi {diff_file_name} {diff_file_1st} 0 1"'
        cmd = cmd.replace(os.sep, '/')
        os.system(cmd)

        ''' Registration from MPRAGE to 1st CHARMED scan using inverse matrix of CHARMED to MPRAGE registration:
        From CHARMED to MPRAGE:'''
        subj_first_charmed = diff_file_name[:-4] + '_1st.nii'
        out_registered = diff_file_name[:-4] + '_1st_reg.nii'
        out_registered_mat = out_registered[:-4] + '.mat'
        options = '-bins 256 -cost normmi -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'
        cmd = f'bash -lc "flirt -ref {subj_mprage} -in {subj_first_charmed} -out {out_registered} -omat {out_registered_mat} {options}"'
        cmd = cmd.replace(os.sep, '/')
        os.system(cmd)

        '''Creation of inverse matrix:  '''
        inv_mat = create_inv_mat(out_registered_mat)

        '''From MPRAGE to CHARMED using the inverse matrix: '''
        out_registered = f'{exp_fol}mprage_reg.nii'
        cmd = f'bash -lc "flirt -in {out_brain} -ref {subj_first_charmed} -out {out_registered} -applyxfm -init {inv_mat}"'
        cmd = cmd.replace(os.sep, '/')
        os.system(cmd)

        '''FAST segmentation:   '''
        fast_seg(out_registered)
