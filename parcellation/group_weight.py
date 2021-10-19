import os
import glob
import nibabel as nib
from weighted_tracts import nodes_labels_yeo7
import numpy as np
from scipy.stats import mode


def calc_mutual_mat(subj_main_folder,atlas_main_folder):
    file_name = 'ADD_by_' + atlas_type
    atlas_labels, mni_atlas_label, idx = atlas_and_idx(atlas_type, atlas_main_folder)
    subj_mat = []
    for subj_fol in glob.glob(f'{subj_main_folder}\*'):
        nii_file_name = os.path.join(subj_fol, file_name + '.nii')
        if os.path.exists(nii_file_name):
            subj_file = nib.load(nii_file_name).get_fdata()

            add_vals = [float(mode(subj_file[atlas_labels == i+1])[0]) for i in idx]
            subj_mat.append(add_vals)
    subj_mat = np.asarray(subj_mat)
    subj_mat[subj_mat == 0] = np.nan
    subj_means = np.nanmean(subj_mat,0)
    subj_medians = np.nanmedian(subj_mat,0)

    return idx,mni_atlas_label,subj_means,subj_medians


def atlas_and_idx(atlas_type, atlas_main_folder):
    mni_atlas_label = f'{atlas_main_folder}{os.sep}{atlas_type}{os.sep}{atlas_type}_atlas.nii'
    atlas_labels = nib.load(mni_atlas_label).get_fdata()
    idx_to_txt = f'{atlas_main_folder}{os.sep}{atlas_type}{os.sep}index2label.txt'
    idx = nodes_labels_yeo7(idx_to_txt)[1]

    return atlas_labels, mni_atlas_label, idx


def weight_atlas_by_add(mni_atlas_file_name,subj_vec,idx):
    labels = nib.load(mni_atlas_file_name).get_fdata()
    weights_dict = {idx[i]: subj_vec[i] for i in range(len(idx))}

    weighted_by_atlas = labels
    for i, weight in weights_dict.items():
        weighted_by_atlas[weighted_by_atlas == i + 1] = weight

    weighted_by_atlas = np.asarray(weighted_by_atlas, dtype='float64')

    return weighted_by_atlas,weights_dict


def save_as_nii(weighted_by_atlas, mni_atlas_file_name, new_file_name, subj_main_folder):

    add_weighted_nii = nib.Nifti1Image(weighted_by_atlas, nib.load(mni_atlas_file_name).affine, nib.load(mni_atlas_file_name).header)
    file_name = os.path.join(subj_main_folder, new_file_name + '.nii')
    nib.save(add_weighted_nii, file_name)


def save_atlas_weights_dict(dict_weight,file_name):
    import json

    with open(file_name,'w') as fp:
        json.dump(dict_weight,fp)


def load_atlas_weights_dict(file_name):
    import json

    with open(file_name, 'r') as fp:
        dict_weight = json.load(fp)

    return dict_weight


if __name__ == '__main__':
    subj_main_folder = 'F:\data\V7\TheBase4Ever'
    atlas_main_folder = r'C:\Users\Admin\my_scripts\aal\yeo'

    atlas_type = 'yeo7_1000'
    idx, mni_atlas_file_name, subj_means, subj_medians = calc_mutual_mat(subj_main_folder,atlas_main_folder)

    weighted_by_means, weights_dict = weight_atlas_by_add(mni_atlas_file_name, subj_means, idx)
    save_as_nii(weighted_by_means,mni_atlas_file_name,'ADD_mean_'+atlas_type,subj_main_folder)
    save_atlas_weights_dict(weights_dict,os.path.join(subj_main_folder,atlas_type+'_mean.json'))

    weighted_by_medians, weights_dict = weight_atlas_by_add(mni_atlas_file_name, subj_medians, idx)
    save_as_nii(weighted_by_medians,mni_atlas_file_name,'ADD_median_'+atlas_type,subj_main_folder)
    save_atlas_weights_dict(weights_dict,os.path.join(subj_main_folder,atlas_type+'_median.json'))

    atlas_type = 'yeo17_1000'
    idx, mni_atlas_file_name, subj_means, subj_medians = calc_mutual_mat(subj_main_folder,atlas_main_folder)

    weighted_by_means, weights_dict = weight_atlas_by_add(mni_atlas_file_name, subj_means, idx)
    save_as_nii(weighted_by_means,mni_atlas_file_name,'ADD_mean_'+atlas_type,subj_main_folder)
    save_atlas_weights_dict(weights_dict,os.path.join(subj_main_folder,atlas_type+'_mean.json'))


    weighted_by_medians, weights_dict = weight_atlas_by_add(mni_atlas_file_name, subj_medians, idx)
    save_as_nii(weighted_by_medians,mni_atlas_file_name,'ADD_median_'+atlas_type,subj_main_folder)
    save_atlas_weights_dict(weights_dict,os.path.join(subj_main_folder,atlas_type+'_median.json'))



