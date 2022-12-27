from parcellation.group_weight import *
import os
import scipy.io as sio

folder_name = r'G:\data\V7\HCP\cm\communities'
th='HistMatch'
atlas = 'yeo7_200'
ncm='SC'
weights = ['Num', 'FA', 'Dist','ADD']

for w in weights:
    communities_file = f'{folder_name}{os.sep}group_division_{atlas}_{w}_{th}_{ncm}.mat'

    mat = sio.loadmat(communities_file)
    val_vec = np.asarray(mat['ciu'])
    idx = np.load(rf'G:\data\V7\HCP\cm\{atlas}_cm_ord_lookup.npy')
#mni_atlas_file_name = r'G:\data\atlases\BNA\BN_Atlas_274_combined_1mm.nii'
#nii_base = r'G:\data\atlases\BNA\MNI152_T1_1mm_brain.nii'
    mni_atlas_file_name = r'G:\data\atlases\yeo\yeo7_200\yeo7_200_atlas.nii'
    nii_base = r'G:\data\atlases\yeo\yeo7_200\Schaefer_template_brain.nii'
    main_subj_folders = r'G:\data\V7\HCP\communities'
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, val_vec, np.sort(idx))
    save_as_nii_aal(weighted_by_atlas, mni_atlas_file_name, nii_base, f'communities_{w}_{atlas}_{th}_{ncm}', main_subj_folders)
