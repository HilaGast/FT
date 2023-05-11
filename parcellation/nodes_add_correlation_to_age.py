import os
import glob
from reading_from_xls.read_details_from_subject_table import *
from parcellation.group_weight import all_subj_add_vals, atlas_and_idx, weight_atlas_by_add, save_as_nii
import numpy as np


def volume_based_var(atlas_type,volume_type, atlas_main_folder, subj_main_folder):

    file_name = f'{volume_type}_by_' + atlas_type
    atlas_labels, mni_atlas_file_name, idx = atlas_and_idx(atlas_type, atlas_main_folder)
    vol_mat = all_subj_add_vals(file_name, atlas_labels, subj_main_folder, idx)
    subj_idx = subj_2_include(subj_main_folder, file_name)

    return vol_mat, mni_atlas_file_name, idx, subj_idx


def age_var(subj_main_folder, subj_idx=None):
    folder_list = glob.glob(f'{subj_main_folder}\*{os.sep}')
    ages=[]
    t1 = SubjTable()
    for f in folder_list:
        subj = str.split(f,os.sep)[4]
        ages.append(t1.find_age_by_scan(subj))

    ages = np.asarray(ages)
    if subj_idx:
        ages = list(ages[subj_idx])
    print(ages)
    return ages


def corr_stats(vol_mat, ages):
    from scipy.stats import pearsonr
    from draw_scatter_fit import remove_outliers_y,remove_nans
    r_vec=[]
    p_vec=[]
    for i in range(np.shape(vol_mat)[1]):
        x = ages
        y = vol_mat[:,i]
        x,y = remove_nans(x,y)
        x,y = remove_outliers_y(x,y)
        x,y = remove_outliers_y(x,y)
        r,p = pearsonr(x,y)
        r_vec.append(r)
        p_vec.append(p)

    return r_vec, p_vec

def subj_2_include(subj_main_folder, file_name):
    if len(subj_main_folder[0]) == 1:
        subj_list = glob.glob(f'{subj_main_folder}{os.sep}*{os.sep}')
    else:
        subj_list = subj_main_folder
    subj_idx = np.zeros(len(subj_list))

    for i in range(len(subj_idx)):
        if file_name+'.nii' in os.listdir(subj_list[i]):
            subj_idx[i] = True
        else:
            subj_idx[i] = False

    return np.bool8(subj_idx)


def multi_comp_correction(r, p):
    from statsmodels.stats.multitest import fdrcorrection_twostage as fdr
    for_comp = [np.asarray(p)>0]
    p_corr_fc = fdr(np.asarray(p)[for_comp],0.05,'bh')[1]
    p_corr = np.asarray(p)
    p_corr[for_comp] = p_corr_fc
    r_th = np.asarray(r)
    r_th[np.asarray(p_corr)>0.05]=0
    r_th = list(r_th)
    p_corr = list(p_corr)

    return r_th, p_corr

if __name__ == '__main__':
    subj_main_folder = 'F:\data\V7\TheBase4Ever'
    atlas_type = 'yeo7_200'
    atlas_main_folder = r'C:\Users\Admin\my_scripts\aal\yeo'
    volume_type = 'ADD'
    vol_mat, mni_atlas_file_name, idx, subj_idx = volume_based_var(atlas_type,volume_type, atlas_main_folder, subj_main_folder)
    num_of_subj = np.shape(vol_mat)[0]

    ages = age_var(subj_main_folder, subj_idx)

    r,p = corr_stats(vol_mat, ages)

    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)

    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_AGE_r_'+atlas_type, subj_main_folder)

    r_th, p_corr = multi_comp_correction(r, p)


    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r_th,idx)

    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_AGE_th_r_'+atlas_type, subj_main_folder)
