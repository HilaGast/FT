from Tractography.connectivity_matrices import ConMat, WeightConMat
import glob, os
from Tractography.files_loading import load_ft


def create_num_add_dist_cm(all_subj_fol, atlas):
    for subj in all_subj_fol:
        streamlines = None
        if not os.path.exists(f'{subj}cm{os.sep}num_{atlas}_cm_ord.npy'):
            if not streamlines:
                streamlines = load_ft(f'{subj}streamlines{os.sep}tracts.trk', f'{subj}diff_corrected.nii')
            cm = ConMat(atlas=atlas, diff_file=f'{subj}diff_corrected.nii', subj_folder=subj,
                            tract_name='tracts.trk',
                            streamlines=streamlines)
            cm.save_cm(fig_name=f'num_{atlas}', mat_type='cm_ord')

        if not os.path.exists(f'{subj}cm{os.sep}add_{atlas}_cm_ord.npy'):
            if not streamlines:
                streamlines = load_ft(f'{subj}streamlines{os.sep}tracts.trk', f'{subj}diff_corrected.nii')
            cm = WeightConMat(weight_by='ADD', atlas=atlas, diff_file=f'{subj}diff_corrected.nii',
                                      subj_folder=subj,
                                      tract_name='tracts.trk',
                                      streamlines=streamlines)
            cm.save_cm(fig_name=f'add_{atlas}', mat_type='cm_ord')

        if not os.path.exists(f'{subj}cm{os.sep}dist_{atlas}_cm_ord.npy'):
            if not streamlines:
                streamlines = load_ft(f'{subj}streamlines{os.sep}tracts.trk', f'{subj}diff_corrected.nii')
            cm = WeightConMat(weight_by='dist', atlas=atlas, diff_file=f'{subj}diff_corrected.nii',
                                      subj_folder=subj,
                                      tract_name='tracts.trk',
                                      streamlines=streamlines)
            cm.save_cm(fig_name=f'dist_{atlas}', mat_type='cm_ord')


if __name__ == '__main__':

    main_fol = 'F:\Hila\TDI\TheBase4Ever'
    all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}')
    all_atlas = ['bnacor', 'yeo7_200']
    for atlas in all_atlas:
        create_num_add_dist_cm(all_subj_fol, atlas)






