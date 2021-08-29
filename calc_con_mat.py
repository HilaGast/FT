from weighted_tracts import *
import glob
import os
import numpy as np
import nibabel as nib

main_folder = r'F:\Hila\balance\eo\before'

for subj_fol in os.listdir(main_folder)[13::]:
    n = subj_fol
    folder_name = os.path.join(main_folder,subj_fol)
    dir_name = folder_name + '\streamlines'
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name, small_delta=15)
    tract_path = glob.glob(dir_name+'*/*_wholebrain_4d_labmask.trk')[0]
    idx = nodes_labels_yeo7(index_to_text_file)[1]
    streamlines = load_ft(tract_path, nii_file)

    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='wholebrain_4d_labmask_yeo7_200_FA',
                               weight_by='_FA')
    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='wholebrain_4d_labmask_yeo7_200',
                               weight_by='_3_2_AxPasi7')
