import glob,os
from reading_from_xls.read_details_from_subject_table import *
from network_analysis.nodes_network_properties import *
from calc_corr_statistics.pearson_r_calc import *
from parcellation.group_weight import atlas_and_idx, weight_atlas_by_add, save_as_nii


if __name__ == "__main__":
    main_subj_folders = r'C:\Users\Admin\Desktop\Language'
    atlas_main_folder = r'C:\Users\Admin\my_scripts\aal\yeo'
    atlas_type = 'yeo7_200'
    atlas_labels, mni_atlas_file_name, idx = atlas_and_idx(atlas_type, atlas_main_folder)

    table1 = SubjTable(r'C:\Users\Admin\Desktop\Language\Subject list - Language.xlsx', 'Sheet1')
    wos1 = []
    lws = []
    n_subj= 0

    for sub in glob.glob(f'{main_subj_folders}{os.sep}*{os.sep}'):
        sn = sub.split(os.sep)[-2]
        num_mat_name = sub + 'non-weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
        if os.path.exists(num_mat_name):

            wos1.append(table1.find_value_by_scan_Language('Word Order Score 1', sn))
            lws.append(table1.find_value_by_scan_Language('Learning words slope', sn))

            n_subj+=1

    nd_num_mat = np.zeros((n_subj,200))
    nd_add_mat = np.zeros((n_subj,200))
    n_subj= 0

    for sub in glob.glob(f'{main_subj_folders}{os.sep}*{os.sep}'):
        sn = sub.split(os.sep)[-2]
        num_mat_name = sub + 'non-weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
        if os.path.exists(num_mat_name):
            num_mat = np.load(num_mat_name)
            nd_num_mat[n_subj,:] = get_node_degree(num_mat)

            add_mat_name = sub + 'weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
            add_mat = np.load(add_mat_name)
            nd_add_mat[n_subj,:] = get_node_degree(add_mat)
            n_subj+=1


    volume_type = 'Num'
    r, p = calc_corr(wos1,nd_num_mat, fdr_correct=True, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_ND-WOS_th_r_'+atlas_type, main_subj_folders)


    r, p = calc_corr(lws,nd_num_mat, fdr_correct=True, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_ND-LWS_th_r_'+atlas_type, main_subj_folders)

    volume_type = 'ADD'

    r, p = calc_corr(wos1,nd_add_mat, fdr_correct=True, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_ND-WOS_th_r_'+atlas_type, main_subj_folders)


    r, p = calc_corr(lws,nd_add_mat, fdr_correct=True, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_ND-LWS_th_r_'+atlas_type, main_subj_folders)