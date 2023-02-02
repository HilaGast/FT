import glob, os
import numpy as np
from HCP_network_analysis.prediction_model.predict_traits_by_networks import create_all_subject_connectivity_matrices, \
    load_trait_vector, pca_for_each_network_different_number_of_components

def whole_brain_matrix(connectivity_matrices):

    labels_networks = ['wb']
    network_mask_dict = {'wb': list(np.arange(0, connectivity_matrices.shape[0]))}
    networks_matrices = {}
    network_mask_vecs = {}
    networks_matrices['wb'] = connectivity_matrices
    network_mask_vecs['wb'] = np.ones(connectivity_matrices.shape[0]*connectivity_matrices.shape[1], dtype = bool)

    return networks_matrices, network_mask_vecs


if __name__ == '__main__':
    from HCP_network_analysis.hcp_cm_parameters import *

    atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
    ncm = ncm_options[0]
    atlas = atlases[0]
    weight_by = weights[2]
    regularization = reg_options[1]
    trait_name =  'CogTotalComp_AgeAdj'
    n_components = 2
    subjects = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
    connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
    network_matrices, network_mask_vec = whole_brain_matrix(connectivity_matrices)
    net_pca, n_components_per_network = pca_for_each_network_different_number_of_components(network_matrices, network_mask_vec, explained_variance_th = 0.2)[0:2]
    trait_vector = load_trait_vector(trait_name, subjects)
    print(weight_by, regularization, ncm)

