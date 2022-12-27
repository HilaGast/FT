from Tractography.connectivity_matrices import ConMat, WeightConMat
from Tractography.fiber_tracking import fiber_tracking_parameters, Tractography
import glob, os
from fsl.file_prep import fast_seg,os_path_2_fsl, create_inv_mat


main_fol = 'G:\data\V7\HCP'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}')

tissue_labels_file_name = 'rMPRAGE_brain_seg.nii'

for fol in all_subj_fol[:15]:

    subj_fol = os_path_2_fsl(f'{fol}{os.sep}')
    dat = f'{subj_fol}data.nii.gz'
    diff_file_1st = dat[:-7] + '_1st'

    # save first corrected diff:
    cmd = fr'bash -lc "fslroi {dat} {diff_file_1st} 0 1"'
    cmd = cmd.replace(os.sep, '/')
    os.system(cmd)

    ''' Registration from MPRAGE to 1st CHARMED scan using inverse matrix of CHARMED to MPRAGE registration:
    From CHARMED to MPRAGE:'''
    subj_first_charmed = dat[:-7] + '_1st.nii'
    out_registered = dat[:-7] + '_1st_reg.nii.gz'
    out_registered_mat = out_registered[:-7] + '.mat'
    subj_mprage = subj_fol + 'MPRAGE_brain.nii'
    options = '-bins 256 -cost normmi -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'
    cmd = f'bash -lc "flirt -ref {subj_mprage} -in {subj_first_charmed} -out {out_registered} -omat {out_registered_mat} {options}"'
    cmd = cmd.replace(os.sep, '/')
    os.system(cmd)

    '''Creation of inverse matrix:  '''
    inv_mat = create_inv_mat(out_registered_mat)

    '''From MPRAGE to CHARMED using the inverse matrix: '''
    out_registered = f'{subj_fol}rMPRAGE_brain.nii'
    out_brain = f'{subj_fol}MPRAGE_brain.nii'
    cmd = f'bash -lc "flirt -in {out_brain} -ref {subj_first_charmed} -out {out_registered} -applyxfm -init {inv_mat}"'
    cmd = cmd.replace(os.sep, '/')
    os.system(cmd)

    fast_seg(out_registered)

    dat = f'{fol}data.nii.gz'

    if not os.path.exists(fol + f'streamlines{os.sep}wb_csd_fa.tck'):
        parameters_dict = fiber_tracking_parameters(max_angle=30, sh_order=6, seed_density=3,
                                                    streamlines_lengths_mm=[50, 500], step_size=1, fa_th=.18)
        tracts = Tractography(fol, 'csd', 'fa', 'wb', parameters_dict, dat,
                              tissue_labels_file_name=tissue_labels_file_name)
        tracts.fiber_tracking()

    if not os.path.exists(f'{fol}cm{os.sep}add_bna_cm_ord.npy'):
        cm = ConMat(atlas='bna', diff_file=dat, subj_folder=fol, tract_name='wb_csd_fa.tck')
        cm.save_cm(fig_name='num_bna', mat_type='cm_ord')

        cm = WeightConMat(weight_by='ADD', atlas='bna', diff_file=dat, subj_folder=fol,
                          tract_name='wb_csd_fa.tck')
        cm.save_cm(fig_name='add_bna', mat_type='cm_ord')

        cm = WeightConMat(weight_by='FA', atlas='bna', diff_file=dat, subj_folder=fol,
                          tract_name='wb_csd_fa.tck')
        cm.save_cm(fig_name='fa_bna', mat_type='cm_ord')




