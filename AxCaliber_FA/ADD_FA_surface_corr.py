import os
import glob
from parcellation.group_weight import all_subj_add_vals, atlas_and_idx, weight_atlas_by_add, save_as_nii
import numpy as np


subj_main_folder = 'F:\data\V7\TheBase4Ever'
atlas_type = 'yeo7_1000'
atlas_main_folder = r'C:\Users\Admin\my_scripts\aal\yeo'

atlas_labels, mni_atlas_file_name, idx = atlas_and_idx(atlas_type, atlas_main_folder)
folder_list = glob.glob(f'{subj_main_folder}\*{os.sep}')

fa_file_name = 'FA_by_' + atlas_type
fa_mat = all_subj_add_vals(fa_file_name, atlas_labels, subj_main_folder, idx)

add_file_name = 'MD_by_' + atlas_type
add_mat = all_subj_add_vals(add_file_name, atlas_labels, subj_main_folder, idx)



from scipy.stats import linregress
r_vec=[]
p_vec=[]
for i in range(np.shape(add_mat)[1]):
    x = fa_mat[:,i]
    y = add_mat[:,i]
    r,p = linregress(x,y)[2:4]
    r_vec.append(r)
    p_vec.append(p)

weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,p_vec,idx)

save_as_nii(weighted_by_atlas, mni_atlas_file_name, r'FA_MD_p_'+atlas_type, subj_main_folder)

weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r_vec,idx)

save_as_nii(weighted_by_atlas, mni_atlas_file_name, r'FA_MD_r_'+atlas_type, subj_main_folder)

r_vec = np.asarray(r_vec)
r_vec[np.asarray(p_vec)>0.05]=0
r_vec = list(r_vec)

weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r_vec,idx)

save_as_nii(weighted_by_atlas, mni_atlas_file_name, r'FA_MD_th_r_'+atlas_type, subj_main_folder)

