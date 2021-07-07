from weighted_tracts import *

subj = all_subj_folders
names = all_subj_names

for s, n in zip(subj[::], names[::]):
    folder_name = subj_folder + s
    dir_name = folder_name + '\streamlines'
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name, small_delta=15)
    tract_path = f'{dir_name}{n}_wholebrain_5d_labmask_msmt.trk'
    idx = nodes_labels_yeo7(index_to_text_file)[1]
    streamlines = load_ft(tract_path, nii_file)

    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='wholebrain_5d_labmask_yeo7_200_FA',
                               weight_by='_FA')
    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='wholebrain_5d_labmask_yeo7_200',
                               weight_by='_3_2_AxPasi7')
