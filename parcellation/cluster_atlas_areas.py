from sklearn.cluster import KMeans
import numpy as np
import os
from parcellation.group_weight import load_atlas_weights_dict, atlas_and_idx, weight_atlas_by_add, save_as_nii

subj_main_folder = 'F:\data\V7\TheBase4Ever'
atlas_main_folder = r'C:\Users\Admin\my_scripts\aal\yeo'
atlas_type = 'yeo17_1000'

file_name = os.path.join(subj_main_folder,atlas_type+'_median.json')
weight_dict = load_atlas_weights_dict(file_name)

add = np.asarray(list(weight_dict.values()))
kmeans = KMeans(n_clusters=17).fit(add.reshape(-1,1))
groups = kmeans.labels_+1

atlas_labels, mni_atlas_label, idx = atlas_and_idx(atlas_type, atlas_main_folder)

weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_label,groups,idx)
new_file_name = os.path.join(subj_main_folder,f'{atlas_type}_group_labels_median')
save_as_nii(weighted_by_atlas,mni_atlas_label,new_file_name,subj_main_folder)
