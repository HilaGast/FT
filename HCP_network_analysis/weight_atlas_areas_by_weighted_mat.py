from parcellation.group_weight import weight_atlas_by_add, save_as_nii_aal
import numpy as np
import os

def average_weighted_nonzero_mat(mat):
    mat[mat==0] = np.nan
    mat_mean = np.nanmean(mat,axis=0)
    mat_mean[np.isnan(mat_mean)] = 0
    return mat_mean



if __name__ == '__main__':
    mni_atlas_file_name = r'G:\data\atlases\yeo\yeo7_200\yeo7_200_atlas.nii'
    nii_base = r'G:\data\atlases\yeo\yeo7_200\Schaefer_template_brain.nii'
    main_subj_folders = r'G:\data\V7\HCP\surfaces_networks_NodeLevel'
    main_mat_folder = r'G:\data\V7\HCP\cm'
    atlas = 'yeo7_200'
    idx = np.load(rf'G:\data\V7\HCP\{atlas}_cm_ord_lookup.npy')
    weights = ['Num', 'Dist', 'FA', 'ADD']
    th = 'HistMatch'
    ncm = 'SC'
    for w in weights:
        mat_name = f'{main_mat_folder}{os.sep}average_{atlas}_{w}_{th}_{ncm}.npy'
        mat = np.load(mat_name)
        weight_vec = average_weighted_nonzero_mat(mat)
        weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, weight_vec, idx)
        save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_NodeAverage_{w}_{th}', main_subj_folders)