import numpy as np
import os, glob

from calc_corr_statistics.pearson_r_calc import calc_corr_mat
from network_analysis.average_weight_per_node import average_val_per_node_per_subject
from parcellation.group_weight import weight_atlas_by_add, save_as_nii

if __name__ == '__main__':
    from HCP_network_analysis.hcp_cm_parameters import *

    atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
    ncm = ncm_options[0]
    atlas = atlases[0]
    regularization = reg_options[2]
    idx = np.load(rf'G:\data\V7\HCP\{atlas}_cm_ord_lookup.npy')
    mni_atlas_file_name = r'G:\data\atlases\yeo\yeo7_200\yeo7_200_atlas.nii'
    output_folder = r'G:\data\V7\HCP\surfaces_networks_NodeLevel\weights_correlations'

    subjects_num = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weights[0]}_{regularization}_{ncm}_cm_ord.npy')
    subjects_fa = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weights[1]}_{regularization}_{ncm}_cm_ord.npy')
    subjects_add = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weights[2]}_{regularization}_{ncm}_cm_ord.npy')

    mat_m_num = average_val_per_node_per_subject(subjects_num)
    mat_m_fa = average_val_per_node_per_subject(subjects_fa)
    mat_m_add = average_val_per_node_per_subject(subjects_add)

    r_num_fa = calc_corr_mat(mat_m_num, mat_m_fa, fdr_correct=True, remove_outliers=False)[0]
    r_num_add = calc_corr_mat(mat_m_num, mat_m_add, fdr_correct=True, remove_outliers=False)[0]
    r_fa_add = calc_corr_mat(mat_m_fa, mat_m_add, fdr_correct=True, remove_outliers=False)[0]

    output_file_name = f'{atlas}_numxfa_{regularization}_{ncm}_pearsonr'
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r_num_fa, idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, output_file_name, output_folder)

    output_file_name = f'{atlas}_numxadd_{regularization}_{ncm}_pearsonr'
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r_num_add, idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, output_file_name, output_folder)

    output_file_name = f'{atlas}_faxadd_{regularization}_{ncm}_pearsonr'
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r_fa_add, idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, output_file_name, output_folder)


