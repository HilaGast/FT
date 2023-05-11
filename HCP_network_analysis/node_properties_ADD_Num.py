import os
import numpy as np
from network_analysis.nodes_network_properties import *
from parcellation.group_weight import *

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
atlas = 'yeo7_200'
th = 'HistMatch'

add_nd=[]
num_nd=[]
fa_nd=[]


add_bc=[]
num_bc=[]
fa_bc=[]




for sl in subj_list:
    subjnum = str.split(sl, os.sep)[2]
    dir_name = f'G:\data\V7\HCP{os.sep}{subjnum}'

    add_cm = np.load(f'{sl}cm{os.sep}{atlas}_ADD_{th}_SC_cm_ord.npy')

    num_cm = np.load(f'{sl}cm{os.sep}{atlas}_Num_{th}_SC_cm_ord.npy')

    fa_cm = np.load(f'{sl}cm{os.sep}{atlas}_FA_{th}_SC_cm_ord.npy')


    add_nd.append(get_node_degree(add_cm))
    add_bc.append(get_node_betweenness_centrality(add_cm))
    num_nd.append(get_node_degree(num_cm))
    num_bc.append(get_node_betweenness_centrality(num_cm))
    fa_nd.append(get_node_degree(fa_cm))
    fa_bc.append(get_node_betweenness_centrality(fa_cm))


num_bc = np.asarray(num_bc)
num_nd = np.asarray(num_nd)

add_bc = np.asarray(add_bc)
add_nd = np.asarray(add_nd)

fa_bc = np.asarray(fa_bc)
fa_nd = np.asarray(fa_nd)



mean_num_bc = np.nanmean(num_bc,axis=0)
mean_num_nd = np.nanmean(num_nd,axis=0)
mean_add_bc = np.nanmean(add_bc,axis=0)
mean_add_nd = np.nanmean(add_nd,axis=0)
mean_fa_bc = np.nanmean(fa_bc, axis=0)
mean_fa_nd = np.nanmean(fa_nd, axis=0)


mni_atlas_file_name = r'G:\data\atlases\yeo\yeo7_200\yeo7_200_atlas.nii'
nii_base = r'G:\data\atlases\yeo\yeo7_200\Schaefer_template_brain.nii'
main_subj_folders = r'G:\data\V7\HCP\surfaces_networks_NodeLevel'
idx = np.load(rf'G:\data\V7\HCP\{atlas}_cm_ord_lookup.npy')



weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_num_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_BetweenessCentrality_Num', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_add_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_BetweenessCentrality_ADD', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_fa_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_BetweenessCentrality_FA', main_subj_folders)


weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_num_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_NodeDegree_Num', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_add_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_NodeDegree_ADD', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_fa_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_NodeDegree_FA', main_subj_folders)

