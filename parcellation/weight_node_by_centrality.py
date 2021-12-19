import os, glob
from parcellation.group_weight import atlas_and_idx, weight_atlas_by_add, save_as_nii
import numpy as np
from network_analysis.nodes_network_properties import get_node_betweenness_centrality


def each_subj_seperate(main_subj_folder,mni_atlas_file_name,idx,atlas_type):
    for sub in glob.glob(f'{main_subj_folder}{os.sep}*{os.sep}'):
        sn = sub.split(os.sep)[-2]
        num_mat_name = sub + 'non-weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
        if os.path.exists(num_mat_name):
            num_mat = np.load(num_mat_name)
            ncb_num = get_node_betweenness_centrality(num_mat)

            add_mat_name = sub + 'weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
            add_mat = np.load(add_mat_name)
            ncb_add = get_node_betweenness_centrality(add_mat)

            weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, ncb_num, idx)
            save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'Num_node-centrality-betweenness_' + atlas_type,
                        sub[:-1])

            weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, ncb_add, idx)
            save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'ADD_node-centrality-betweenness_' + atlas_type,
                        sub[:-1])



def grouped_together(main_subj_folder,mni_atlas_file_name,idx,atlas_type):

    ncb_num=[]
    ncb_add=[]
    for sub in glob.glob(f'{main_subj_folder}{os.sep}*{os.sep}'):
        sn = sub.split(os.sep)[-2]
        num_mat_name = sub + 'non-weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
        if os.path.exists(num_mat_name):
            num_mat = np.load(num_mat_name)
            ncb_num.append(get_node_betweenness_centrality(num_mat))

            add_mat_name = sub + 'weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
            add_mat = np.load(add_mat_name)
            ncb_add.append(get_node_betweenness_centrality(add_mat))

    ncb_num = np.asarray(ncb_num)
    ncb_add = np.asarray(ncb_add)
    ncb_num[ncb_num==0] = np.nan
    ncb_add[ncb_add==0] = np.nan

    ncb_num_mean = np.nanmean(ncb_num,axis=0)
    ncb_add_mean = np.nanmean(ncb_add,axis=0)
    ncb_num_mean[np.isnan(ncb_num_mean)] = 0
    ncb_add_mean[np.isnan(ncb_add_mean)] = 0

    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, ncb_num_mean, idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'Num_node-centrality-betweenness_' + atlas_type,
                        main_subj_folder)

    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, ncb_add_mean, idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'ADD_node-centrality-betweenness_' + atlas_type,
                        main_subj_folder)

if __name__ == "__main__":
    main_subj_folder = 'F:\data\V7\TheBase4Ever'
    atlas_type = 'yeo7_200'
    atlas_main_folder = r'C:\Users\Admin\my_scripts\aal\yeo'
    volume_type = 'ADD'
    atlas_labels, mni_atlas_file_name, idx = atlas_and_idx(atlas_type, atlas_main_folder)
    grouped_together(main_subj_folder,mni_atlas_file_name,idx,atlas_type)







