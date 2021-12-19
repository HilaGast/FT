import os, glob
from network_analysis.nodes_network_properties import *
from parcellation.nodes_add_correlation_to_age import age_var
import numpy as np
import nibabel as nib
from calc_corr_statistics.pearson_r_calc import *
from parcellation.group_weight import atlas_and_idx, weight_atlas_by_add, save_as_nii

def mean_dict_vals(d1):
    for k,v in d1.items():
        d1[k] = np.nanmean(v)

    return d1


def save_dict_as_nii(d1, mni_atlas_file_name, new_file_name, subj_main_folder):
    weighted_by_atlas = nib.load(mni_atlas_file_name).get_fdata()
    for i, weight in d1.items():
        weighted_by_atlas[weighted_by_atlas == i + 1] = weight

    weighted_by_atlas = np.asarray(weighted_by_atlas, dtype='float64')

    add_weighted_nii = nib.Nifti1Image(weighted_by_atlas, nib.load(mni_atlas_file_name).affine, nib.load(mni_atlas_file_name).header)
    file_name = os.path.join(subj_main_folder, new_file_name + '.nii')
    nib.save(add_weighted_nii, file_name)




if __name__ == '__main__':

    main_subj_folders = 'F:\data\V7\TheBase4Ever'
    atlas_main_folder = r'C:\Users\Admin\my_scripts\aal\yeo'
    atlas_type = 'yeo7_200'
    atlas_labels, mni_atlas_file_name, idx = atlas_and_idx(atlas_type, atlas_main_folder)

    eff_num_dict = {}
    eff_add_dict = {}
    subj_idx = []

    for sub in glob.glob(f'{main_subj_folders}{os.sep}*{os.sep}'):

        num_mat_name = sub + 'non-weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
        if os.path.exists(num_mat_name):
            subj_idx.append(True)
            num_mat = np.load(num_mat_name)
            eff_num = (get_local_efficiency(cm=num_mat))
            eff_num_dict = merge_dict(eff_num_dict, eff_num)

            add_mat_name = sub + 'weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
            add_mat = np.load(add_mat_name)
            eff_add = get_local_efficiency(cm=add_mat)
            eff_add_dict = merge_dict(eff_add_dict, eff_add)

        else:
            subj_idx.append(False)

    ages = age_var(main_subj_folders, subj_idx)

    eff_num_mat =np.zeros((len(ages),len(eff_num_dict)))
    eff_add_mat = np.zeros((len(ages),len(eff_add_dict)))
    for k in eff_num_dict.keys():
        eff_num_mat[:, k] = eff_num_dict[k]

    for k in eff_add_dict.keys():
        eff_add_mat[:, k] = eff_add_dict[k]

    mni_atlas_file_name = r'C:\Users\Admin\my_scripts\aal\yeo\yeo7_200\yeo7_200_atlas.nii'

    volume_type = 'Num'
    r, p = calc_corr(ages,eff_num_mat, fdr_correct=True, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LocEff-AGE_th_r_'+atlas_type, main_subj_folders)

    volume_type = 'ADD'
    r, p = calc_corr(ages,eff_add_mat, fdr_correct=True, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LocEff-AGE_th_r_'+atlas_type, main_subj_folders)




