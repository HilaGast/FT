from weighted_tracts import *
import glob,os



main_folder = r'F:\data\V7\TheBase4Ever'
for subj_fol in glob.glob(main_folder + f'{os.sep}*{os.sep}'):
    sub = subj_fol.split(os.sep)[-2]
    n = os.sep+sub.split('_')[3]
    nii_file = load_dwi_files(subj_fol, small_delta=15)[5]
    dir_name = subj_fol + '\streamlines'
    #tract_path = f'{dir_name}{n}_wholebrain_5d_labmask_msmt.trk'
    tract_path = f'{dir_name}{os.sep}lab_wholebrain_5d_labmask_msmt.trk'

    try:

        streamlines = load_ft(tract_path, nii_file)

    except FileNotFoundError:

        try:
            tract_path = f'{dir_name}{os.sep}lab_wholebrain_5d_labmask_msmt.trk'
            streamlines = load_ft(tract_path, nii_file)
        except FileNotFoundError:

            continue

    weighted_connectivity_matrix_mega(streamlines, subj_fol, fig_type='wholebrain_5d_labmask_bna',
                                      weight_by='3_2_AxPasi7', atlas='bna_cor')



