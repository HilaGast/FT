 import glob, os
from parcellation.add_weighted_gm_mask import GM_mask
from parcellation.nodes_add_correlation_to_age import subj_2_include
from parcellation.group_weight import atlas_and_idx, weight_atlas_by_add, save_as_nii
import nibabel as nib
import numpy as np


def vol_by_atlas(wt, main_folder,atlas_name, atlas_main_folder):
    for subj_fol in glob.glob(main_folder + f'{os.sep}e*{os.sep}*{os.sep}*'):

        if not os.path.exists(os.path.join(subj_fol,'streamlines')):
            print('Could not find streamlines file')
            continue

        file_name = f'{wt}_by_{atlas_name}'
        if os.path.exists(os.path.join(subj_fol, file_name + '.nii')):
            print(f'Done with \n {file_name} \n {subj_fol} \n')
        else:
            if 'FA' in wt:
                subj_mask = GM_mask(subj_fol=subj_fol, atlas_name=atlas_name, weight_by='FA')
            elif 'MD' in wt:
                subj_mask = GM_mask(subj_fol=subj_fol, atlas_name=atlas_name, weight_by='MD')
            else:
                subj_mask = GM_mask(subj_fol=subj_fol, atlas_name=atlas_name, tractography_type='_wholebrain_4d_labmask', atlas_main_folder=atlas_main_folder)
            subj_mask.weight_gm_by_add()
            subj_mask.save_weighted_gm_mask(file_name=file_name)
            print(f'Done with \n {file_name} \n {subj_fol} \n')


def volume_based_var(atlas_type,volume_type, atlas_main_folder, group_folders):

    file_name = f'{volume_type}_by_' + atlas_type
    atlas_labels, mni_atlas_file_name, idx = atlas_and_idx(atlas_type, atlas_main_folder)
    vol_mat =  group_subj_add_vols(file_name, atlas_labels, group_folders, idx)
    subj_idx = subj_2_include(group_folders, file_name)

    return vol_mat, mni_atlas_file_name, idx, subj_idx


def group_subj_add_vols(file_name, atlas_labels, group_folders, idx):
    from scipy.stats import mode
    subj_mat = []
    for subj_fol in group_folders:
        nii_file_name = os.path.join(subj_fol, file_name + '.nii')
        if os.path.exists(nii_file_name):
            subj_file = nib.load(nii_file_name).get_fdata()

            add_vals = [float(mode(subj_file[atlas_labels == i+1])[0]) for i in idx]
            subj_mat.append(add_vals)
    subj_mat = np.asarray(subj_mat)

    return subj_mat


def multi_t_test(mat1, mat2, fdr_correct=True):
    from scipy.stats import ttest_rel
    t, p = ttest_rel(mat1, mat2, axis=0, nan_policy='omit')

    if fdr_correct:
        from calc_corr_statistics.pearson_r_calc import multi_comp_correction
        t, p, t_th = multi_comp_correction(t, p)
    else:
        import copy
        t_th = copy.deepcopy(t)
        t_th[p>0.05] = 0

    return t, p ,t_th


if __name__ == '__main__':
    wt ='ADD'
    main_folder = r'F:\data\balance'
    atlas_name = 'bna'
    atlas_main_folder = r'F:\data\atlases\BNA'

    #vol_by_atlas(wt, main_folder, atlas_name, atlas_main_folder)

    before_subj = glob.glob(main_folder + f'{os.sep}e*{os.sep}before{os.sep}*')
    after_subj = glob.glob(main_folder + f'{os.sep}e*{os.sep}after{os.sep}*')

    before_vol_mat, mni_atlas_file_name, idx, subj_idx =  volume_based_var(atlas_name, wt, atlas_main_folder, before_subj)
    after_vol_mat, mni_atlas_file_name, idx, subj_idx =  volume_based_var(atlas_name, wt, atlas_main_folder, after_subj)


    t, p ,t_th = multi_t_test(before_vol_mat, after_vol_mat, fdr_correct=False)

    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,t_th,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'before_vs_after_t_th_'+atlas_name, main_folder)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,t,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'before_vs_after_t_'+atlas_name, main_folder)









