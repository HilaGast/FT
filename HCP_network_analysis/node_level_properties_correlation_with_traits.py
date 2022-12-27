import os, glob
import pandas as pd
from draw_scatter_fit import *
from calc_corr_statistics.pearson_r_calc import *
from parcellation.group_weight import weight_atlas_by_add, save_as_nii_aal
from network_analysis.nodes_network_properties import *

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
idx = np.load(r'G:\data\V7\HCP\bna_cm_ord_lookup.npy')
mni_atlas_file_name = r'G:\data\atlases\BNA\BN_Atlas_274_combined_1mm.nii'
nii_base = r'G:\data\atlases\yeo\yeo7_1000\Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii'
main_subj_folders = r'G:\data\V7\HCP'


add_nodes=[]
num_nodes=[]
fa_nodes=[]
num_fa_nodes = []
num_add_nodes=[]
#cols = ['CogTotalComp_AgeAdj','CogFluidComp_AgeAdj','CogCrystalComp_AgeAdj','ReadEng_AgeAdj','PicVocab_AgeAdj','ProcSpeed_AgeAdj','IWRD_TOT','IWRD_RTC','Language_Task_Acc','Language_Task_Story_Acc']
cols = ['Noise_Comp','Odor_AgeAdj','PainInterf_Tscore','Taste_AgeAdj','Mars_Final']
#col = cols[3]
table1 = pd.read_csv('G:\data\V7\HCP\HCP_behavioural_data.csv')
atlas = 'bna'
node_property = 'nd' # from ['le', 'nd', 'bc', 'cc']

for sl in subj_list:

    subjnum = str.split(sl, os.sep)[-2]
    dir_name = f'G:\data\V7\HCP{os.sep}{subjnum}'

    #col_val.append(float(table1[col][table1['Subject']==int(subjnum)].values))
    add_cm = np.load(f'{dir_name}{os.sep}cm{os.sep}add_{atlas}_cm_ord_histmatch_th.npy')
    num_cm = np.load(f'{dir_name}{os.sep}cm{os.sep}num_{atlas}_cm_ord_histmatch_th.npy')
    fa_cm = np.load(f'{dir_name}{os.sep}cm{os.sep}fa_{atlas}_cm_ord_histmatch_th.npy')

    if node_property == 'nd':
        subj_add_prop = get_node_degree(add_cm)
        add_nodes.append(subj_add_prop)
        subj_num_prop = get_node_degree(num_cm)
        num_nodes.append(subj_num_prop)
        subj_fa_prop = get_node_degree(fa_cm)
        fa_nodes.append(subj_fa_prop)
        subj_num_fa_prop = get_node_degree(num_cm*fa_cm)
        num_fa_nodes.append(subj_num_fa_prop)
        subj_num_add_prop = get_node_degree(num_cm * add_cm)
        num_add_nodes.append(subj_num_add_prop)


    if node_property == 'le':
        subj_add_prop = get_local_efficiency(add_cm)
        add_nodes.append(subj_add_prop)
        subj_num_prop = get_local_efficiency(num_cm)
        num_nodes.append(subj_num_prop)
        subj_fa_prop = get_local_efficiency(fa_cm)
        fa_nodes.append(subj_fa_prop)
        subj_num_fa_prop = get_local_efficiency(num_cm * fa_cm)
        num_fa_nodes.append(subj_num_fa_prop)
        subj_num_add_prop = get_local_efficiency(num_cm * add_cm)
        num_add_nodes.append(subj_num_add_prop)


    if node_property == 'bc':
        subj_add_prop = get_node_betweenness_centrality(add_cm)
        add_nodes.append(subj_add_prop)
        subj_num_prop = get_node_betweenness_centrality(num_cm)
        num_nodes.append(subj_num_prop)
        subj_fa_prop = get_node_betweenness_centrality(fa_cm)
        fa_nodes.append(subj_fa_prop)
        subj_num_fa_prop = get_node_betweenness_centrality(num_cm * fa_cm)
        num_fa_nodes.append(subj_num_fa_prop)
        subj_num_add_prop = get_node_betweenness_centrality(num_cm * add_cm)
        num_add_nodes.append(subj_num_add_prop)


    if node_property == 'cc':
        subj_add_prop = get_node_clustering_coefficient(add_cm)
        add_nodes.append(subj_add_prop)
        subj_num_prop = get_node_clustering_coefficient(num_cm)
        num_nodes.append(subj_num_prop)
        subj_fa_prop = get_node_clustering_coefficient(fa_cm)
        fa_nodes.append(subj_fa_prop)
        subj_num_fa_prop = get_node_clustering_coefficient(num_cm * fa_cm)
        num_fa_nodes.append(subj_num_fa_prop)
        subj_num_add_prop = get_node_clustering_coefficient(num_cm * add_cm)
        num_add_nodes.append(subj_num_add_prop)


num_nodes = np.asarray(num_nodes)
add_nodes = np.asarray(add_nodes)
fa_nodes = np.asarray(fa_nodes)
num_fa_nodes = np.asarray(num_fa_nodes)
num_add_nodes = np.asarray(num_add_nodes)

for col in cols:
    col_val = []
    for sl in subj_list:
        subjnum = str.split(sl, os.sep)[-2]
        col_val.append(float(table1[col][table1['Subject']==int(subjnum)].values))

    r,p = calc_corr(col_val,add_nodes, fdr_correct=True)
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
    save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_{node_property}_correlation_ADD_{col}', main_subj_folders)

    r,p = calc_corr(col_val,num_nodes, fdr_correct=True)
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
    save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_{node_property}_correlation_Num_{col}', main_subj_folders)

    r,p = calc_corr(col_val,fa_nodes, fdr_correct=True)
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
    save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_{node_property}_correlation_FA_{col}', main_subj_folders)

    r,p = calc_corr(col_val,num_fa_nodes, fdr_correct=True)
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
    save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_{node_property}_correlation_NumxFA_{col}', main_subj_folders)

    r,p = calc_corr(col_val,num_add_nodes, fdr_correct=True)
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
    save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_{node_property}_correlation_NumxADD_{col}', main_subj_folders)
