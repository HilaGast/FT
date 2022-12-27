import os, glob
import pandas as pd
from draw_scatter_fit import *
from calc_corr_statistics.pearson_r_calc import *
from parcellation.group_weight import weight_atlas_by_add, save_as_nii_aal

shortlist = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')


ages=[]
add=[]

table1 = pd.read_csv('G:\data\V7\HCP\HCP_demographic_data.csv')

atlas = 'bna'
for sl in shortlist:
    dir_name = sl[:-1]
    subj_number = sl.split(os.sep)[-2]


    add_cm = np.load(f'{dir_name}{os.sep}cm{os.sep}add_{atlas}_cm_ord_histmatch_th.npy')

    num_cm = np.load(f'{dir_name}{os.sep}cm{os.sep}num_bna_cm_ord_histmatch_th.npy')

    mutual = np.nansum(add_cm*num_cm, axis=0)

    add.append(mutual/np.nansum(num_cm, axis=0))

    ages.append(int(table1['Age_in_Yrs'][table1['Subject']==int(subj_number)].values))

add = np.asarray(add)

r,p = calc_corr(ages,add, fdr_correct=False)

idx = np.load(r'G:\data\V7\HCP\bna_cm_ord_lookup.npy')
mni_atlas_file_name = r'G:\data\atlases\BNA\BN_Atlas_274_combined_1mm.nii'
nii_base = r'G:\data\atlases\yeo\yeo7_1000\Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii'

main_subj_folders = r'G:\data\V7\HCP'

weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'bna_correlation_ADD_Age', main_subj_folders)


