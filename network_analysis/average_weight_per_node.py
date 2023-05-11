import numpy as np
import os

from HCP_network_analysis.weight_atlas_areas_by_weighted_mat import average_weighted_nonzero_mat


def average_val_per_node_per_subject(subjects, calc_type='mean'):
    for i, s in enumerate(subjects):
        subj_mat = np.load(s)
        subj_vec = average_weighted_nonzero_mat(subj_mat)
        if i==0:
            all_subj_mat = np.zeros((subj_vec.shape[0], len(subjects)))
        all_subj_mat[:,i] = subj_vec
    all_subj_mat[all_subj_mat == 0] = np.nan

    return all_subj_mat
