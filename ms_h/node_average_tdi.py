from parcellation.group_weight import weight_atlas_by_add, save_as_nii_aal
import numpy as np
import os

def average_weighted_nonzero_mat(mat):
    mat[mat==0] = np.nan
    mat_mean = np.nanmean(mat,axis=0)
    mat_mean[np.isnan(mat_mean)] = 0
    return mat_mean



if __name__ == '__main__':
    mni_atlas_file_name = r'G:\data\atlases\BNA\newBNA_Labels.nii'
    nii_base = r'G:\data\atlases\BNA\MNI152_T1_1mm_brain.nii'
    main_subj_folders = r'F:\Hila\TDI\siemens\group_cm'
    main_mat_folder = r'F:\Hila\TDI\siemens\group_cm'
    atlas = 'bnacor'
    idx = np.load(rf'F:\Hila\TDI\siemens\group_cm\{atlas}_cm_ord_lookup.npy')
    mat_type = 'time_th30'
    exp = 'D60d11'
    group = 'ms'
    mat_name = f'{main_mat_folder}{os.sep}median_{mat_type}_{atlas}_{exp}_{group}.npy'
    mat = np.load(mat_name)
    weight_vec = average_weighted_nonzero_mat(mat)
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, weight_vec, idx)
    save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_NodeAverage_{mat_type}_{exp}_{group}', main_subj_folders)