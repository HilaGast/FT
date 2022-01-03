import os, glob
from calc_corr_statistics.pearson_r_calc import calc_corr_mat
from network_analysis.nodes_network_properties import *
from parcellation.group_weight import weight_atlas_by_add, save_as_nii_aal

shortlist = glob.glob(f'F:\data\V7\HCP\*{os.sep}')
#shortlist = ['H:\\HCP\\123925\\', 'H:\\HCP\\193441\\', 'H:\\HCP\\304727\\', 'H:\\HCP\\204016\\', 'H:\\HCP\\297655\\', 'H:\\HCP\\186848\\', 'H:\\HCP\\180432\\', 'H:\\HCP\\555348\\', 'H:\\HCP\\283543\\', 'H:\\HCP\\181232\\', 'H:\\HCP\\387959\\', 'H:\\HCP\\613538\\', 'H:\\HCP\\153227\\', 'H:\\HCP\\414229\\', 'H:\\HCP\\753251\\', 'H:\\HCP\\115017\\', 'H:\\HCP\\248339\\', 'H:\\HCP\\168947\\', 'H:\\HCP\\349244\\', 'H:\\HCP\\547046\\', 'H:\\HCP\\199453\\', 'H:\\HCP\\110411\\', 'H:\\HCP\\886674\\', 'H:\\HCP\\113922\\', 'H:\\HCP\\151223\\', 'H:\\HCP\\130518\\', 'H:\\HCP\\380036\\', 'H:\\HCP\\671855\\', 'H:\\HCP\\286650\\', 'H:\\HCP\\158136\\', 'H:\\HCP\\194443\\', 'H:\\HCP\\154734\\', 'H:\\HCP\\106824\\', 'H:\\HCP\\880157\\', 'H:\\HCP\\111009\\', 'H:\\HCP\\299154\\', 'H:\\HCP\\599469\\', 'H:\\HCP\\530635\\', 'H:\\HCP\\609143\\', 'H:\\HCP\\130922\\', 'H:\\HCP\\245333\\', 'H:\\HCP\\513130\\', 'H:\\HCP\\154229\\', 'H:\\HCP\\178647\\', 'H:\\HCP\\192439\\', 'H:\\HCP\\395251\\', 'H:\\HCP\\112314\\', 'H:\\HCP\\168240\\', 'H:\\HCP\\150928\\', 'H:\\HCP\\884064\\', 'H:\\HCP\\180129\\', 'H:\\HCP\\792867\\', 'H:\\HCP\\517239\\', 'H:\\HCP\\106319\\', 'H:\\HCP\\481951\\', 'H:\\HCP\\628248\\', 'H:\\HCP\\522434\\', 'H:\\HCP\\123420\\', 'H:\\HCP\\129331\\', 'H:\\HCP\\397760\\', 'H:\\HCP\\951457\\', 'H:\\HCP\\157336\\', 'H:\\HCP\\104820\\', 'H:\\HCP\\118730\\', 'H:\\HCP\\701535\\', 'H:\\HCP\\550439\\', 'H:\\HCP\\176744\\', 'H:\\HCP\\871762\\', 'H:\\HCP\\715647\\', 'H:\\HCP\\151627\\', 'H:\\HCP\\707749\\', 'H:\\HCP\\195849\\', 'H:\\HCP\\206727\\', 'H:\\HCP\\336841\\', 'H:\\HCP\\197348\\', 'H:\\HCP\\188145\\', 'H:\\HCP\\192237\\', 'H:\\HCP\\191437\\', 'H:\\HCP\\139435\\', 'H:\\HCP\\663755\\', 'H:\\HCP\\158843\\', 'H:\\HCP\\118124\\', 'H:\\HCP\\469961\\', 'H:\\HCP\\894673\\', 'H:\\HCP\\872158\\', 'H:\\HCP\\765864\\', 'H:\\HCP\\203923\\', 'H:\\HCP\\134829\\', 'H:\\HCP\\111716\\', 'H:\\HCP\\742549\\', 'H:\\HCP\\144933\\', 'H:\\HCP\\286347\\', 'H:\\HCP\\130417\\', 'H:\\HCP\\911849\\', 'H:\\HCP\\263436\\', 'H:\\HCP\\102614\\', 'H:\\HCP\\101107\\', 'H:\\HCP\\120414\\', 'H:\\HCP\\210415\\', 'H:\\HCP\\214019\\', 'H:\\HCP\\180937\\', 'H:\\HCP\\113215\\', 'H:\\HCP\\121719\\', 'H:\\HCP\\248238\\', 'H:\\HCP\\727654\\', 'H:\\HCP\\352132\\', 'H:\\HCP\\627549\\', 'H:\\HCP\\158540\\', 'H:\\HCP\\212217\\', 'H:\\HCP\\307127\\', 'H:\\HCP\\378756\\', 'H:\\HCP\\138837\\', 'H:\\HCP\\381038\\', 'H:\\HCP\\117021\\', 'H:\\HCP\\877168\\', 'H:\\HCP\\134021\\', 'H:\\HCP\\159138\\']


add_ne_dict={}
num_ne_dict={}


for sl in shortlist:
    dir_name = sl[:-1]
    #subjnum = str.split(sl, os.sep)[2]
    #dir_name = f'F:\data\V7\HCP{os.sep}{subjnum}'

    add_cm = np.load(f'{dir_name}{os.sep}cm_add.npy')
    add_ne = get_local_efficiency(add_cm)
    add_ne_dict = merge_dict(add_ne_dict, add_ne)

    num_cm = np.load(f'{dir_name}{os.sep}cm_num.npy')
    num_ne = get_local_efficiency(num_cm)
    num_ne_dict = merge_dict(num_ne_dict, num_ne)



eff_num_mat =np.zeros((len(num_ne_dict[0]),len(num_ne_dict)))
eff_add_mat = np.zeros((len(add_ne_dict[0]),len(add_ne_dict)))
for k in num_ne_dict.keys():
    eff_num_mat[:, k] = num_ne_dict[k]

for k in add_ne_dict.keys():
    eff_add_mat[:, k] = add_ne_dict[k]

idx = np.load(r'F:\data\V7\HCP\cm_num_lookup.npy')
mni_atlas_file_name = r'F:\data\atlases\aal300\AAL150_fixed.nii'
nii_base = r'F:\data\atlases\yeo\yeo7_1000\Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii'

main_subj_folders = r'F:\data\V7\HCP'

r, p = calc_corr_mat(eff_num_mat,eff_add_mat,fdr_correct=True)
weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'aal300_LocEff_correlation_ADD_Num_fdr', main_subj_folders)

r, p = calc_corr_mat(eff_num_mat,eff_add_mat,fdr_correct=False)
weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'aal300_LocEff_correlation_ADD_Num', main_subj_folders)


eff_add_mean = np.nanmean(eff_add_mat, axis=0)
weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, eff_add_mean, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'aal300_LocEff_ADD', main_subj_folders)

eff_num_mean = np.nanmean(eff_num_mat, axis=0)
weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, eff_num_mean, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'aal300_LocEff_Num', main_subj_folders)