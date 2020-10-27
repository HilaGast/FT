from weighted_tracts import *

subj = all_subj_folders
names = all_subj_names
fig_type = 'wholebrain_4d_labmask_FA_DTI'
fig_name = r'weighted_mega_' + fig_type + '.npy'

for s, n in zip(subj, names):
    folder_name = subj_folder + s
    dir_name = folder_name + '\streamlines'


    if fig_name in os.listdir(folder_name):
        print('Moving on!')
        continue

    if 'rdti_fa.nii' not in os.listdir(folder_name):
        print('No matching volume file')
        continue

    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)

    tract_path = f'{dir_name}{n}_wholebrain_4d_labmask.trk'
    streamlines = load_ft(tract_path, nii_file)
    print(f'streamline loaded for {n}')

    #weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='wholebrain_4d_labmask_FA',
    #                              weight_by='_FA')
    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type=fig_type,
                                  weight_by='rdti_fa')
