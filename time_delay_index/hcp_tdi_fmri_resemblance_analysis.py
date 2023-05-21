import glob, os
import numpy as np

from Tractography.group_analysis import create_all_subject_connectivity_matrices, norm_matrices

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
atlases = ['yeo7_100']
tdi_names = []
fmri_names = []
for atlas in atlases:
    for s in subj_list:
        tdi_file_name = f'{s}cm{os.sep}{atlas}_time_th3_Org_SC_cm_ord.npy'
        fmri_file_name = f'{s}cm{os.sep}{atlas}_fmri_Org_SC_cm_ord.npy'
        if os.path.exists(tdi_file_name) and os.path.exists(fmri_file_name):
            tdi_names.append(tdi_file_name)
            fmri_names.append(fmri_file_name)
    # Load
    tdi_mats = create_all_subject_connectivity_matrices(tdi_names)
    fmri_mats = create_all_subject_connectivity_matrices(fmri_names)

    # opposite tdi?


    # Normalize
    tdi_norm = norm_matrices(tdi_mats, norm_type='z-score')
    fmri_norm = norm_matrices(fmri_mats, norm_type='fisher')

            # Find distance\similarity between tdi and fmri

            # Save the distance\similarity

    # Plot the distance\similarity

    # Save the plot






