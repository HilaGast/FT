
from dipy.io.streamline import load_trk
from single_fascicle_vizualization import streamline_mean_fascicle_value_weighted
from weighted_tracts import load_dwi_files,weighting_streamlines
import numpy as np
import os


main_folder = r'F:\Hila\balance\eo\after'

bundle_name = 'SCP'
file_bundle_name = bundle_name + r'_mct01rt20_4d'

for subj in os.listdir(main_folder):
    folder_name = os.path.join(main_folder, subj)
    full_bund_name = f'{subj}_{file_bundle_name}'
    bundle_file_name = rf'{folder_name}\streamlines\{full_bund_name}.trk'

    if not os.path.exists(bundle_file_name):
        print('Moving on!')
        continue

    bundle = load_trk(bundle_file_name, "same", bbox_valid_check=False)
    bundle = bundle.streamlines
    bvec_file = load_dwi_files(folder_name)[6]
    mean_vols = weighting_streamlines(folder_name,bundle,bvec_file,weight_by='AxPasi7')

    mean_bundle = np.nanmean(mean_vols)
    median_bundle = np.nanmedian(mean_vols)

    print(f'{subj} mean value for {bundle_name} : {mean_bundle:.2f}\n {subj} median value for {bundle_name} : {median_bundle:.2f} \n')
