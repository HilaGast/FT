import glob, os
import numpy as np

from HCP_network_analysis.weight_atlas_areas_by_weighted_mat import average_weighted_nonzero_mat
from network_analysis.nodes_network_properties import get_local_efficiency, merge_dict
from parcellation.group_weight import weight_atlas_by_add, save_as_nii_aal


def divide_group_left_right(subjects):
    """
    Divide subjects into left and right handers
    :param subjects: list of subjects
    :return: left and right handers
    """
    import pandas as pd
    left = []
    right = []
    table1 = pd.read_csv('G:\data\V7\HCP\HCP_demographic_data.csv')
    for s in subjects:
        subj_id = s.split('\\')[-3]
        handness = float(table1['Handedness'][table1['Subject']==int(subj_id)].values)
        if handness >= 50:
            right.append(s)
        elif handness <= -50:
            left.append(s)
    return left, right

def ne_dict(subjects):
    """
    Get local efficiency dictionary
    :param subjects: list of subjects
    :return: local efficiency dictionary
    """
    import numpy as np

    ne_dict = {}
    for s in subjects:
        cm = np.load(s)
        ne = get_local_efficiency(cm, return_dict=True)
        ne_dict = merge_dict(ne_dict, ne)
    return ne_dict


def save_average_loceff_as_nii(ne_dict, mni_atlas_file_name, nii_base, idx, output_file_name, output_folder):
    """
    Save average local efficiency as nii file
    :param ne_dict: local efficiency dictionary
    :param mni_atlas_file_name: atlas file name
    :param nii_base: nii file to base the output on
    :param output_file_name: output file name
    :return: None
    """

    eff_mat = np.zeros((len(ne_dict[0]), len(ne_dict)))
    for k in ne_dict.keys():
        eff_mat[:, k] = ne_dict[k]

    eff_mean = np.nanmean(eff_mat, axis=0)
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, eff_mean, idx)
    save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, output_file_name, output_folder)


def average_left_right_mat(subjects, output_folder, output_file_name, calc_type='mean'):

    for i, s in enumerate(subjects):
        subj_mat = np.load(s)
        if i==0:
            all_subj_mat = np.zeros((subj_mat.shape[0], subj_mat.shape[1], len(subjects)))
        all_subj_mat[:,:,i] = subj_mat
    all_subj_mat[all_subj_mat == 0] = np.nan
    mat = all_subj_mat.copy()
    mat_m = np.zeros((all_subj_mat.shape[0], all_subj_mat.shape[1]))
    for r in range(all_subj_mat.shape[0]):
        for c in range(all_subj_mat.shape[0]):
            rc = mat[r,c,:]
            if np.count_nonzero(~np.isnan(rc)) < len(subjects)/5:
                mat_m[r, c] = 0
            else:
                if calc_type == 'mean':
                    mat_m[r,c]= np.nanmean(rc[rc>0])
    fig_name = os.path.join(output_folder, output_file_name)
    np.save(fig_name,mat_m)
    return mat_m


if __name__ == '__main__':
    from HCP_network_analysis.hcp_cm_parameters import *

    atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
    ncm = ncm_options[0]
    atlas = atlases[0]
    weight_by = weights[2]
    regularization = reg_options[1]
    idx = np.load(rf'G:\data\V7\HCP\{atlas}_cm_ord_lookup.npy')
    mni_atlas_file_name = r'G:\data\atlases\yeo\yeo7_200\yeo7_200_atlas.nii'
    nii_base = r'G:\data\atlases\yeo\yeo7_200\Schaefer_template_brain.nii'
    output_folder = r'G:\data\V7\HCP\surfaces_networks_NodeLevel'
    subjects = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
    subjects_l, subjects_r = divide_group_left_right(subjects)
    print(f'Left handers: {len(subjects_l)}')
    print(f'Right handers: {len(subjects_r)}')

    #add_ne_dict_l = ne_dict(subjects_l)
    #add_ne_dict_r = ne_dict(subjects_r)
    # output_file_name = f'{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord_lefthand_LocEff'
    # save_average_loceff_as_nii(add_ne_dict_l, mni_atlas_file_name, nii_base, idx, output_file_name, output_folder)
    # output_file_name = f'{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord_righthand_LocEff'
    # save_average_loceff_as_nii(add_ne_dict_r, mni_atlas_file_name, nii_base, idx, output_file_name, output_folder)


    output_folder_cm = r'G:\data\V7\HCP\cm'
    output_file_name = f'average_{atlas}_{weight_by}_{regularization}_{ncm}_lefthand'
    mat_m_l = average_left_right_mat(subjects_l, output_folder_cm, output_file_name, calc_type='mean')
    output_file_name = f'average_{atlas}_{weight_by}_{regularization}_{ncm}_righthand'
    mat_m_r = average_left_right_mat(subjects_r, output_folder_cm, output_file_name, calc_type='mean')

    vec_l = average_weighted_nonzero_mat(mat_m_l)
    output_file_name = f'{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord_lefthand_NodeAvg'
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, vec_l, idx)
    save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, output_file_name, output_folder)

    vec_r = average_weighted_nonzero_mat(mat_m_r)
    output_file_name = f'{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord_righthand_NodeAvg'
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, vec_r, idx)
    save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, output_file_name, output_folder)




