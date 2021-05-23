from weighted_tracts import *

s = all_subj_folders[10]
n = all_subj_names[10]
folder_name = subj_folder + s
dir_name = folder_name + '\streamlines'
gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name, small_delta=15)
lab_labels_index = nodes_by_index_general(folder_name, atlas='yeo7_200')[0]




print('Starting tractography sh=6, d=3')
model_fit = create_csd_model(data, gtab, white_matter, sh_order=6)
den = 3
seeds = create_seeds(folder_name, lab_labels_index, affine, use_mask=False, mask_type='cc', den=den)
tract_file_name = "_wholebrain_3d_labmask_sh6_cmc_pft.trk"
streamlines = create_streamlines(model_fit, seeds, affine, folder_name=folder_name, classifier_type="cmc")
save_ft(folder_name, n, streamlines, nii_file, file_name=tract_file_name)

