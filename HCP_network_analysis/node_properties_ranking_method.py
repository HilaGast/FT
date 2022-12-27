import os
import numpy as np
from network_analysis.nodes_network_properties import *
from parcellation.group_weight import *
from scipy.stats import rankdata

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

add_le=[]
num_le=[]
fa_le=[]
num_fa_le = []
num_add_le=[]
th = '_val_th'
for sl in subj_list:
    subjnum = str.split(sl, os.sep)[2]
    dir_name = f'G:\data\V7\HCP{os.sep}{subjnum}'

    add_cm = np.load(f'{sl}cm{os.sep}add_bna_cm_ord{th}.npy')

    num_cm = np.load(f'{sl}cm{os.sep}num_bna_cm_ord{th}.npy')

    fa_cm = np.load(f'{sl}cm{os.sep}fa_bna_cm_ord{th}.npy')

    #mask_mat = num_cm>3
    #norm_add = add_cm*mask_mat
    #norm_fa = fa_cm*mask_mat
    #norm_num = num_cm*mask_mat

    subj_add_nd = rankdata(get_node_degree(add_cm))
    add_nd.append(subj_add_nd)
    subj_add_bc = rankdata(get_node_betweenness_centrality(add_cm))
    add_bc.append(subj_add_bc)
    subj_add_le = rankdata(get_local_efficiency(add_cm))
    add_le.append(subj_add_le)

    subj_num_nd = rankdata(get_node_degree(num_cm))
    num_nd.append(subj_num_nd)
    subj_num_bc = rankdata(get_node_betweenness_centrality(num_cm))
    num_bc.append(subj_num_bc)
    subj_num_le = rankdata(get_local_efficiency(num_cm))
    num_le.append(subj_num_le)

    subj_fa_nd = rankdata(get_node_degree(fa_cm))
    fa_nd.append(subj_fa_nd)
    subj_fa_bc = rankdata(get_node_betweenness_centrality(fa_cm))
    fa_bc.append(subj_fa_bc)
    subj_fa_le = rankdata(get_local_efficiency(fa_cm))
    fa_le.append(subj_fa_le)


    num_fa_nd.append(rankdata(subj_num_nd+subj_fa_nd))
    num_fa_bc.append(rankdata(subj_num_bc+subj_fa_bc))
    num_fa_le.append(rankdata(subj_num_le + subj_fa_le))
    num_add_nd.append(rankdata(subj_num_nd+subj_add_nd))
    num_add_bc.append(rankdata(subj_num_bc+subj_add_bc))
    num_add_le.append(rankdata(subj_num_le+subj_add_le))

num_bc = np.asarray(num_bc)
num_nd = np.asarray(num_nd)
num_le = np.asarray(num_le)

add_bc = np.asarray(add_bc)
add_nd = np.asarray(add_nd)
add_le = np.asarray(add_le)

fa_bc = np.asarray(fa_bc)
fa_nd = np.asarray(fa_nd)
fa_le = np.asarray(fa_le)

num_fa_bc = np.asarray(num_fa_bc)
num_fa_nd = np.asarray(num_fa_nd)
num_fa_le = np.asarray(num_fa_le)

num_add_bc = np.asarray(num_add_bc)
num_add_nd = np.asarray(num_add_nd)
num_add_le = np.asarray(num_add_le)

mean_num_bc = np.nanmean(num_bc,axis=0)
mean_num_nd = np.nanmean(num_nd,axis=0)
mean_num_le = np.nanmean(num_le, axis=0)

mean_add_bc = np.nanmean(add_bc,axis=0)
mean_add_nd = np.nanmean(add_nd,axis=0)
mean_add_le = np.nanmean(add_le,axis=0)

mean_fa_bc = np.nanmean(fa_bc, axis=0)
mean_fa_nd = np.nanmean(fa_nd, axis=0)
mean_fa_le = np.nanmean(fa_le, axis=0)

mean_numfa_bc = np.nanmean(num_fa_bc,axis=0)
mean_numfa_nd = np.nanmean(num_fa_nd,axis=0)
mean_numfa_le = np.nanmean(num_fa_le, axis=0)

mean_numadd_bc = np.nanmean(num_add_bc,axis=0)
mean_numadd_nd = np.nanmean(num_add_nd,axis=0)
mean_numadd_le = np.nanmean(num_add_le, axis=0)

idx = np.load(r'G:\data\V7\HCP\bna_cm_ord_lookup.npy')
mni_atlas_file_name = r'G:\data\atlases\BNA\BN_Atlas_274_combined_1mm.nii'
nii_base = r'G:\data\atlases\BNA\MNI152_T1_1mm_brain.nii'

main_subj_folders = r'G:\data\V7\HCP'


weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_num_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_BetweenessCentrality_Num(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_add_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_BetweenessCentrality_ADD(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_fa_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_BetweenessCentrality_FA(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_numfa_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_BetweenessCentrality_NumxFA(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_numadd_bc, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_BetweenessCentrality_NumxADD(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_num_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_NodeDegree_Num(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_add_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_NodeDegree_ADD(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_fa_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_NodeDegree_FA(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_numfa_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_NodeDegree_NumxFA(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_numadd_nd, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_NodeDegree_NumxADD(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_num_le, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_LocEff_Num(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_add_le, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_LocEff_ADD(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_fa_le, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_LocEff_FA(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_numfa_le, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_LocEff_NumxFA(rank){th}', main_subj_folders)

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, mean_numadd_le, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_LocEff_NumxADD(rank){th}', main_subj_folders)