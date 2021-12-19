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
    eff_num_dict = {}
    eff_add_dict = {}
    wos1 = []
    lws = []
    n_subj= 0

    for sub in glob.glob(f'{main_subj_folders}{os.sep}*{os.sep}'):
        sn = sub.split(os.sep)[-2]
        num_mat_name = sub + 'non-weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
        if os.path.exists(num_mat_name):
            n_subj+=1
            num_mat = np.load(num_mat_name)
            eff_num = (get_local_efficiency(cm=num_mat))
            eff_num_dict = merge_dict(eff_num_dict, eff_num)

            add_mat_name = sub + 'weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
            add_mat = np.load(add_mat_name)
            eff_add = get_local_efficiency(cm=add_mat)
            eff_add_dict = merge_dict(eff_add_dict, eff_add)

            wos1.append(table1.find_value_by_scan_Language('Word Order Score 1', sn))
            lws.append(table1.find_value_by_scan_Language('Learning words slope', sn))

    eff_num_mat =np.zeros((n_subj,len(eff_num_dict)))
    eff_add_mat = np.zeros((n_subj,len(eff_add_dict)))
    for k in eff_num_dict.keys():
        eff_num_mat[:, k] = eff_num_dict[k]

    for k in eff_add_dict.keys():
        eff_add_mat[:, k] = eff_add_dict[k]

    volume_type = 'Num'
    r, p = calc_corr(wos1,eff_num_mat, fdr_correct=False, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LocEff-WOS_th_r_'+atlas_type, main_subj_folders)


    r, p = calc_corr(lws,eff_num_mat, fdr_correct=False, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LocEff-LWS_th_r_'+atlas_type, main_subj_folders)

    volume_type = 'ADD'

    r, p = calc_corr(wos1,eff_add_mat, fdr_correct=False, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LocEff-WOS_th_r_'+atlas_type, main_subj_folders)


    r, p = calc_corr(lws,eff_add_mat, fdr_correct=False, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LocEff-LWS_th_r_'+atlas_type, main_subj_folders)