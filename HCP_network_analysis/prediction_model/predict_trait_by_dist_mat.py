import glob, os
import numpy as np
from HCP_network_analysis.prediction_model.predict_traits_by_networks import create_all_subject_connectivity_matrices, \
    load_trait_vector, pca_for_each_network_different_number_of_components


def from_whole_brain_to_dist_net(dist_mat,connectivity_matrices, k):
    group_label_dist_mat = define_dist_kmeans_groups(dist_mat, k)

    dist_net_matrices = {}
    dist_net_mask_vecs = {}

    for group in range(1, k+1):
        group_mask = np.zeros(dist_mat.shape, dtype = bool)
        group_mask[group_label_dist_mat == group] = True
        dist_net_mask_vecs[str(group)] = group_mask.flatten()
        group_mask = np.repeat(group_mask[:,:,np.newaxis], connectivity_matrices.shape[2], axis = 2)
        dist_net_matrices[str(group)] = connectivity_matrices*group_mask


    return dist_net_matrices, dist_net_mask_vecs


def define_dist_kmeans_groups(dist_mat, k):
    from sklearn.cluster import KMeans
    dist_vals = dist_mat[dist_mat > 0]
    group_label_dist_mat = dist_mat.copy()
    kmeans = KMeans(n_clusters=k,  init=np.asarray([[1],[5],[10],[20],[40],[70],[110],[160]]), random_state=0).fit(dist_vals.reshape(-1, 1))
    group_predict = kmeans.predict(dist_vals.reshape(-1, 1))+1
    group_label_dist_mat[group_label_dist_mat > 0] = group_predict
    return group_label_dist_mat


if __name__ == '__main__':
    from HCP_network_analysis.hcp_cm_parameters import *

    atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
    ncm = ncm_options[0]
    atlas = atlases[0]
    weight_by = weights[2]
    regularization = reg_options[1]
    trait_name =  'CogTotalComp_AgeAdj'
    k=3
    n_components = 2
    subjects = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
    dist_mat = np.load(f'G:\data\V7\HCP\cm{os.sep}average_{atlas}_Dist_{regularization}_{ncm}.npy')

    connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
    dist_net_matrices, dist_net_mask_vec = from_whole_brain_to_dist_net(dist_mat,connectivity_matrices, k)

    dist_pca, n_components_per_network = pca_for_each_network_different_number_of_components(dist_net_matrices, dist_net_mask_vec, explained_variance_th = 0.2)[0:2]

    trait_vector = load_trait_vector(trait_name, subjects)
    print(weight_by, regularization, ncm)



