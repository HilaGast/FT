
from dipy.io.streamline import load_trk
from all_subj import *
from single_fascicle_vizualization import streamline_mean_fascicle_value_weighted
from weighted_tracts import load_dwi_files
import numpy as np
import os

main_folder = subj_folder
names = all_subj_names
folders = all_subj_folders

bundle_name = 'F_L_R_mct001rt20'

for f,n in zip(folders,names):
    folder_name = f'{main_folder}{f}'
    bundle_file_name = rf'{folder_name}\streamlines{n}_{bundle_name}.trk'

    if not os.path.exists(bundle_file_name):
        print('Moving on!')
        continue

    bundle = load_trk(bundle_file_name, "same", bbox_valid_check=False)
    bundle = bundle.streamlines
    nii_file = load_dwi_files(folder_name)[5]
    streamlines, vec_vols = streamline_mean_fascicle_value_weighted(folder_name, n, nii_file,bundle_name,
                                                                    bundle, weight_by='_AxPasi')
    mean_bundle = np.mean(vec_vols)

    print(f'{n} mean value for {bundle_name} : {mean_bundle:.2f}')