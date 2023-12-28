from skimage.measure import regionprops
import os
import nibabel as nib
import numpy as np


def find_labels_file(cm_name_path):

    path_parts = cm_name_path.split(os.sep)
    atlas_name = path_parts[-1]

    if 'bna_' in atlas_name:
        labels_file = 'rBN_Atlas_274_combined_1mm.nii'
    elif 'yeo7_200' in atlas_name:
        labels_file = 'ryeo7_200_atlas.nii'
    elif 'yeo7_100' in atlas_name:
        labels_file = 'ryeo7_100_atlas.nii'
    elif 'bnacor' in atlas_name:
        labels_file = 'rnewBNA_Labels.nii'

    else:
        print('could not find labels file')

    labels_file_path = f'{os.path.dirname(os.path.dirname(cm_name_path))}{os.sep}{labels_file}'

    return labels_file_path


def find_labels_centroids(labels_file_path):
    labels_img = nib.load(labels_file_path).get_fdata()
    props = regionprops(labels_img.astype(np.int))
    label_ctd = {}
    for n in props:
        label_ctd[n.label] = n.centroid

    return label_ctd


def euc_dist_mat(label_ctd, cm):
    euc_mat = np.zeros(cm.shape)
    for k1,v1 in label_ctd.items():
        for k2,v2 in label_ctd.items():
            if k1 != k2:
                euc_mat[k1-1,k2-1] = np.linalg.norm(np.asarray(v2) - np.asarray(v1))
    euc_mat = (euc_mat+euc_mat.T)/2

    return euc_mat
