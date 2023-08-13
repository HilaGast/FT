from Tractography.connectivity_matrices import ConMat, WeightConMat
import glob, os
from Tractography.files_loading import load_ft


def create_num_add_dist_cm(all_subj_fol, atlas, tract_file_name='tracts.tck',cm_name_extras=''):
    for subj in all_subj_fol:
        streamlines = None
        if not os.path.exists(f'{subj}cm{os.sep}{atlas}_Num{cm_name_extras}_cm_ord.npy'):
            if not streamlines:
                streamlines = load_ft(f'{subj}streamlines{os.sep}{tract_file_name}', f'{subj}diff_corrected.nii')
            cm = ConMat(atlas=atlas, diff_file=f'{subj}diff_corrected.nii', subj_folder=subj,
                            tract_name=tract_file_name,
                            streamlines=streamlines)
            cm.save_cm(fig_name=f'{atlas}_Num{cm_name_extras}', mat_type=f'cm_ord')

        if not os.path.exists(f'{subj}cm{os.sep}{atlas}_ADD{cm_name_extras}_cm_ord.npy'):
            if not streamlines:
                streamlines = load_ft(f'{subj}streamlines{os.sep}{tract_file_name}', f'{subj}diff_corrected.nii')
            cm = WeightConMat(weight_by='ADD', atlas=atlas, diff_file=f'{subj}diff_corrected.nii',
                                      subj_folder=subj,
                                      tract_name=tract_file_name,
                                      streamlines=streamlines)
            cm.save_cm(fig_name=f'{atlas}_ADD{cm_name_extras}', mat_type=f'cm_ord')

        if not os.path.exists(f'{subj}cm{os.sep}{atlas}_Dist{cm_name_extras}_cm_ord.npy'):
            if not streamlines:
                streamlines = load_ft(f'{subj}streamlines{os.sep}{tract_file_name}', f'{subj}diff_corrected.nii')
            cm = WeightConMat(weight_by='dist', atlas=atlas, diff_file=f'{subj}diff_corrected.nii',
                                      subj_folder=subj,
                                      tract_name=tract_file_name,
                                      streamlines=streamlines)
            cm.save_cm(fig_name=f'{atlas}_Dist{cm_name_extras}', mat_type=f'cm_ord')


if __name__ == '__main__':

    main_fol = 'F:\Hila\TDI\TheBase4Ever'
    all_subj_fol = glob.glob(f'{main_fol}{os.sep}*[0-9]{os.sep}')
    all_atlas = ['yeo7_100', 'yeo7_200','bnacor']
    for atlas in all_atlas:
        create_num_add_dist_cm(all_subj_fol, atlas,tract_file_name='tracts_short.tck',cm_name_extras='_shortth')






