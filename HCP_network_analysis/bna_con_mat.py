from HCP_network_analysis.cc_violin_HCP import load_tck

tck_file_name = f'{s}HCP_tracts.tck'
nii_file_name = sl + 'data.nii'
affine = nib.load(nii_file_name).affine
streamlines = load_tck(tck_file_name, nii_file_name)
weighted_connectivity_matrix_mega(streamlines, folder_name, fig_type = 'whole brain', weight_by='3_2_AxPasi7',atlas='bna')
