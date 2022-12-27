import os, glob
from calc_corr_statistics.pearson_r_calc import calc_corr_mat
from network_analysis.nodes_network_properties import *
from parcellation.group_weight import weight_atlas_by_add, save_as_nii_aal

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
atlas = 'yeo7_200'
th = 'HistMatch'
add_ne_dict={}
num_ne_dict={}
dist_ne_dict={}
fa_ne_dict={}


for sl in subj_list:
    dir_name = sl[:-1]
    #subjnum = str.split(sl, os.sep)[2]
    #dir_name = f'F:\data\V7\HCP{os.sep}{subjnum}'

    add_cm = np.load(f'{sl}cm{os.sep}{atlas}_ADD_{th}_SC_cm_ord.npy')
    add_ne = get_local_efficiency(add_cm, return_dict=True)
    add_ne_dict = merge_dict(add_ne_dict, add_ne)

    num_cm = np.load(f'{sl}cm{os.sep}{atlas}_Num_{th}_SC_cm_ord.npy')
    num_ne = get_local_efficiency(num_cm, return_dict=True)
    num_ne_dict = merge_dict(num_ne_dict, num_ne)

    fa_cm = np.load(f'{sl}cm{os.sep}{atlas}_FA_{th}_SC_cm_ord.npy')
    fa_ne = get_local_efficiency(fa_cm, return_dict=True)
    fa_ne_dict = merge_dict(fa_ne_dict, fa_ne)

    dist_cm = np.load(f'{sl}cm{os.sep}{atlas}_Dist_{th}_SC_cm_ord.npy')
    dist_ne = get_local_efficiency(dist_cm, return_dict=True)
    dist_ne_dict = merge_dict(dist_ne_dict, dist_ne)


eff_num_mat =np.zeros((len(num_ne_dict[0]),len(num_ne_dict)))
eff_add_mat = np.zeros((len(add_ne_dict[0]),len(add_ne_dict)))
eff_fa_mat = np.zeros((len(fa_ne_dict[0]),len(fa_ne_dict)))
eff_dist_mat = np.zeros((len(dist_ne_dict[0]),len(dist_ne_dict)))

for k in num_ne_dict.keys():
    eff_num_mat[:, k] = num_ne_dict[k]

for k in add_ne_dict.keys():
    eff_add_mat[:, k] = add_ne_dict[k]

for k in fa_ne_dict.keys():
    eff_fa_mat[:, k] = fa_ne_dict[k]

for k in dist_ne_dict.keys():
    eff_dist_mat[:, k] = dist_ne_dict[k]

idx = np.load(rf'G:\data\V7\HCP\{atlas}_cm_ord_lookup.npy')
# mni_atlas_file_name = r'F:\data\atlases\aal300\AAL150_fixed.nii'
# nii_base = r'F:\data\atlases\yeo\yeo7_1000\Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii'
mni_atlas_file_name = r'G:\data\atlases\yeo\yeo7_200\yeo7_200_atlas.nii'
nii_base = r'G:\data\atlases\yeo\yeo7_200\Schaefer_template_brain.nii'
main_subj_folders = r'G:\data\V7\HCP\surfaces_networks_NodeLevel'


eff_add_mean = np.nanmean(eff_add_mat, axis=0)
weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, eff_add_mean, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_LocEff_ADD', main_subj_folders)

eff_num_mean = np.nanmean(eff_num_mat, axis=0)
weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, eff_num_mean, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_LocEff_Num', main_subj_folders)

eff_fa_mean = np.nanmean(eff_fa_mat, axis=0)
weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, eff_fa_mean, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_LocEff_FA', main_subj_folders)

eff_dist_mean = np.nanmean(eff_dist_mat, axis=0)
weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, eff_dist_mean, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'{atlas}_LocEff_Dist', main_subj_folders)