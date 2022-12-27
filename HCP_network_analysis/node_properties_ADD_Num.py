import os
import numpy as np
from network_analysis.nodes_network_properties import *
from parcellation.group_weight import *

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')

add_nd=[]
num_nd=[]
fa_nd=[]
num_fa_nd = []
num_add_nd=[]

add_bc=[]
num_bc=[]
fa_bc=[]
num_fa_bc = []
num_add_bc=[]



for sl in subj_list:
    subjnum = str.split(sl, os.sep)[2]
    dir_name = f'G:\data\V7\HCP{os.sep}{subjnum}'

    add_cm = np.load(f'{sl}cm{os.sep}add_bna_cm_ord.npy')
    #add_nd.append(get_node_degree(add_cm))
    #add_bc.append(get_node_betweenness_centrality(add_cm))

    num_cm = np.load(f'{sl}cm{os.sep}num_bna_cm_ord.npy')
    #num_nd.append(get_node_degree(num_cm))
    #num_bc.append(get_node_betweenness_centrality(num_cm))

    fa_cm = np.load(f'{sl}cm{os.sep}fa_bna_cm_ord.npy')
    #fa_nd.append(get_node_degree(fa_cm))
    #fa_bc.append(get_node_betweenness_centrality(fa_cm))


    mask_mat = num_cm>3
    norm_add = add_cm*mask_mat
    norm_fa = fa_cm*mask_mat
    norm_num = np.log(num_cm)*mask_mat
    add_num = norm_num * norm_add
    fa_num = norm_num * norm_fa

    add_nd.append(get_node_degree(norm_add))
    add_bc.append(get_node_betweenness_centrality(norm_add))
    num_nd.append(get_node_degree(norm_num))
    num_bc.append(get_node_betweenness_centrality(norm_num))
    fa_nd.append(get_node_degree(norm_fa))
    fa_bc.append(get_node_betweenness_centrality(norm_fa))
    num_fa_nd.append(get_node_degree(fa_num))
    num_fa_bc.append(get_node_betweenness_centrality(fa_num))
    num_add_nd.append(get_node_degree(add_num))
    num_add_bc.append(get_node_betweenness_centrality(add_num))

num_bc = np.asarray(num_bc)
num_nd = np.asarray(num_nd)

add_bc = np.asarray(add_bc)
add_nd = np.asarray(add_nd)

fa_bc = np.asarray(fa_bc)
fa_nd = np.asarray(fa_nd)

num_fa_bc = np.asarray(num_fa_bc)
num_fa_nd = np.asarray(num_fa_nd)

num_add_bc = np.asarray(num_add_bc)
num_add_nd = np.asarray(num_add_nd)

mean_num_bc = np.nanmean(num_bc,axis=0)
mean_num_nd = np.nanmean(num_nd,axis=0)
mean_add_bc = np.nanmean(add_bc,axis=0)
mean_add_nd = np.nanmean(add_nd,axis=0)
mean_fa_bc = np.nanmean(fa_bc, axis=0)
mean_fa_nd = np.nanmean(fa_nd, axis=0)
mean_numfa_bc = np.nanmean(num_fa_bc,axis=0)
mean_numfa_nd = np.nanmean(num_fa_nd,axis=0)
mean_numadd_bc = np.nanmean(num_add_bc,axis=0)
mean_numadd_nd = np.nanmean(num_add_nd,axis=0)

idx = np.load(r'G:\data\V7\HCP\bna_cm_ord_lookup.npy')
mni_atlas_file_name = r'G:\data\atlases\BNA\newBNA_Labels.nii'
nii_base = r'G:\data\atlases\BNA\MNI152_T1_1mm_brain.nii'

main_subj_folders = r'G:\data\V7\HCP'


weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_num_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_BetweenessCentrality_Num(lognorm)', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_add_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_BetweenessCentrality_ADD(norm)', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_fa_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_BetweenessCentrality_FA(norm)', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_numfa_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_BetweenessCentrality_NumxFA(norm)', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_numadd_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_BetweenessCentrality_NumxADD(norm)', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_num_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_NodeDegree_Num(lognorm)', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_add_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_NodeDegree_ADD(norm)', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_fa_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_NodeDegree_FA(norm)', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_numfa_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_NodeDegree_NumxFA(norm)', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_numadd_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_NodeDegree_NumxADD(norm)', main_subj_folders)