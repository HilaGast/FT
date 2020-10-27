from weighted_tracts import *

subj = all_subj_folders
names = all_subj_names

for s, n in zip(subj[::], names[::]):
    folder_name = subj_folder + s
    dir_name = folder_name + '\streamlines'
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
    idx = nodes_labels_aal3(index_to_text_file)[1]
    tract_path = f'{dir_name}{n}_lrg_4d_labmask.trk'
    streamlines = load_ft(tract_path, nii_file)
    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='lrg_4d_labmask_FA_aal3',
                                      weight_by='_FA')
    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='lrg_4d_labmask_aal3',
                                      weight_by='_AxPasi')
    tract_path = f'{dir_name}{n}_med_4d_labmask.trk'
    streamlines = load_ft(tract_path, nii_file)
    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='med_4d_labmask_FA_aal3',
                                      weight_by='_FA')
    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='med_4d_labmask_aal3',
                                      weight_by='_AxPasi')
    tract_path = f'{dir_name}{n}_sml_4d_labmask.trk'
    streamlines = load_ft(tract_path, nii_file)
    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='sml_4d_labmask_FA_aal3',
                                      weight_by='_FA')
    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='sml_4d_labmask_aal3',
                                      weight_by='_AxPasi')