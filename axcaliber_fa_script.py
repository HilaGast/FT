from weighted_tracts import *

subj = all_subj_folders
names = all_subj_names

for s, n in zip(subj, names):
    folder_name = subj_folder + s
    dir_name = folder_name + '\streamlines'
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
    tract_path = f'{dir_name}{n}_wholebrain_4d_labmask.trk'
    streamlines = load_ft(tract_path, nii_file)
    print(f'streamline loaded for {n}')
    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='wholebrain_4d_labmask',
                                  weight_by='_FA')
