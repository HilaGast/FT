import glob
import os
import numpy as np
import nibabel as nib

main_folder = r'F:\Hila\balance'

for subj_fol in glob.glob(main_folder + r'*/e*/*/*/'):
    mat_num = np.load(os.path.join(subj_fol, 'non-weighted_wholebrain_4d_labmask_yeo7_200_nonnorm.npy'))
    mat_add = np.load(os.path.join(subj_fol, 'weighted_wholebrain_4d_labmask_yeo7_200_nonnorm.npy'))
    mat_fa = np.load(os.path.join(subj_fol, 'weighted_wholebrain_4d_labmask_yeo7_200_FA_nonnorm.npy'))
    atlas_file = nib.load(os.path.join(subj_fol,'ryeo7_200_atlas.nii')).get_fdata()

    num_sum = (np.sum(mat_num) + np.sum(np.eye(mat_num.shape[0], mat_num.shape[1]) * mat_num)) / 2
    norm_mat_num = mat_num * 100 / num_sum # %of all streamlines in specific subject

    # now I want to normalize my reults to the atlas area:
    lab_labels_index = [labels for labels in atlas_file]
    lab_labels_index = np.asarray(lab_labels_index, dtype='int')

    atlas_labels = np.unique(lab_labels_index)

    labels_vec=[]
    for li in atlas_labels[atlas_labels!=0]:
        labels_vec.append(np.sum(lab_labels_index==li))

    labels_mat = np.repeat(labels_vec,200,axis=0).reshape(200,200)
    sizes_mat = (labels_mat+labels_mat.transpose())/2

    norm_mat_num = norm_mat_num/sizes_mat

    norm_mat_add = (mat_add - np.min(mat_add[mat_add > 0])) / (np.max(mat_add) - np.min(mat_add[mat_add > 0]))
    norm_mat_add[mat_add == 0] = 0

    norm_mat_fa = (mat_fa - np.min(mat_fa[mat_fa > 0])) / (np.max(mat_fa) - np.min(mat_fa[mat_fa > 0]))
    norm_mat_fa[mat_fa == 0] = 0

    norm_mat_add_num = norm_mat_num*norm_mat_add
    norm_mat_fa_num = norm_mat_num*norm_mat_fa

    np.save(os.path.join(subj_fol,'norm_num_mat.npy'),norm_mat_num)
    np.save(os.path.join(subj_fol, 'norm_add_mat.npy'), norm_mat_add)
    np.save(os.path.join(subj_fol, 'norm_fa_mat.npy'), norm_mat_fa)
    np.save(os.path.join(subj_fol, 'norm_num-add_mat.npy'), norm_mat_add_num)
    np.save(os.path.join(subj_fol, 'norm_num-fa_mat.npy'), norm_mat_fa_num)



