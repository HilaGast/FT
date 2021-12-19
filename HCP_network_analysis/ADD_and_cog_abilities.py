import os, glob
import pandas as pd
from draw_scatter_fit import *
from calc_corr_statistics.pearson_r_calc import *
from parcellation.group_weight import weight_atlas_by_add, save_as_nii_aal

shortlist = glob.glob(f'F:\data\V7\HCP\*{os.sep}')

total=[]
fluid=[]
crystal=[]
add=[]

table1 = pd.read_csv('F:\data\V7\HCP\HCP_behavioural_data.csv')

idx = np.load(r'F:\data\V7\HCP\cm_num_lookup.npy')
mni_atlas_file_name = r'F:\data\atlases\aal300\AAL150_fixed.nii'
nii_base = r'F:\data\atlases\yeo\yeo7_1000\Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii'

main_subj_folders = r'F:\data\V7\HCP'

for sl in shortlist:
    dir_name = sl[:-1]
    subj_number = sl.split(os.sep)[-2]


    add_cm = np.load(f'{dir_name}{os.sep}cm_add.npy')

    num_cm = np.load(f'{dir_name}{os.sep}cm_num.npy')

    mutual = np.nansum(add_cm*num_cm, axis=0)

    add.append(mutual/np.nansum(num_cm, axis=0))

    total.append(float(table1['CogTotalComp_AgeAdj'][table1['Subject']==int(subj_number)].values))
    fluid.append(float(table1['CogFluidComp_AgeAdj'][table1['Subject']==int(subj_number)].values))
    crystal.append(float(table1['CogCrystalComp_AgeAdj'][table1['Subject']==int(subj_number)].values))

add = np.asarray(add)

r,p = calc_corr(total,add, fdr_correct=False)
weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'aal300_correlation_ADD_total', main_subj_folders)


r,p = calc_corr(fluid,add, fdr_correct=False)
weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'aal300_correlation_ADD_fluid', main_subj_folders)

r,p = calc_corr(crystal,add, fdr_correct=False)
weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'aal300_correlation_ADD_crystal', main_subj_folders)

