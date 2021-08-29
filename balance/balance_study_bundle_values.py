
from dipy.io.streamline import load_trk
from weighted_tracts import load_dwi_files,weighting_streamlines
import numpy as np
import os
import glob
import pandas as pd


bundles = ['MCP', 'SCP', 'ICP_L', 'ICP_R', 'OR_L', 'OR_R', 'CST_L', 'CST_R']
main_folder = r'F:\Hila\balance\eo'
val_dict={}
for bundle_name in bundles:
    file_bundle_name = bundle_name + r'_mct01rt20_4d'

    for group in glob.glob(main_folder + r'*/*/'):
        vec_mean=[]
        vec_median=[]
        for subj in os.listdir(group):
            folder_name = os.path.join(group, subj)
            full_bund_name = f'{subj}_{file_bundle_name}'
            bundle_file_name = rf'{folder_name}\streamlines\{full_bund_name}.trk'

            if not os.path.exists(bundle_file_name):
                print('Moving on!')
                vec_mean.append(np.nan)
                vec_median.append(np.nan)
                continue

            bundle = load_trk(bundle_file_name, "same", bbox_valid_check=False)
            bundle = bundle.streamlines
            bvec_file = load_dwi_files(folder_name)[6]
            mean_vols = weighting_streamlines(folder_name,bundle,bvec_file,weight_by='ADD_along_streamlines_WMmasked')

            vec_mean.append(np.nanmean(mean_vols))
            vec_median.append(np.nanmedian(mean_vols))
        val_dict[bundle_name+'_mean_'+str.split(group,os.sep)[-2]] = vec_mean
        val_dict[bundle_name + '_median_'+str.split(group,os.sep)[-2]] = vec_median

table_vals = pd.DataFrame(val_dict,index=os.listdir(group))

table_vals.to_excel(os.path.join(main_folder,'values_avaraged_volume.xlsx'))




