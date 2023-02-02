from HCP_network_analysis.prediction_model.predict_traits_by_networks import *

def from_whole_brain_to_hemispheres(connectivity_matrices, atlas_index_labels):

    labels_file = open(atlas_index_labels, 'r', errors='ignore')
    labels_name = labels_file.readlines()
    labels_file.close()
    labels_networks = ['LH', 'RH']
    hemi_mask_dict = create_dict_of_hemisphere_indices(labels_name, labels_networks)
    hemi_matrices, hemi_mask_vecs = create_hemisphere_matrices(connectivity_matrices, hemi_mask_dict)

    return hemi_matrices, hemi_mask_vecs

def create_dict_of_hemisphere_indices(labels_name, labels_networks):
    hemi_mask_dict = {}
    for hemi in labels_networks:
        hemi_mask_dict[hemi] = []
    for l in labels_name:
        label_parts = l.split('\t')
        hemi_mask_dict[label_parts[1].split('_')[1]].append(int(label_parts[0]) - 1)
    return hemi_mask_dict

def create_hemisphere_matrices(connectivity_matrices, hemi_mask_dict):
    hemi_matrices = {}
    all_masks = []
    hemi_mask_vecs = {}

    for hemi in hemi_mask_dict.keys():
        hemi_mask = np.zeros(connectivity_matrices.shape, dtype = bool)
        for r in hemi_mask_dict[hemi]:
            for c in hemi_mask_dict[hemi]:
                hemi_mask[r, c, :] = True
        hemi_matrices[hemi] = connectivity_matrices*hemi_mask
        hemi_mask_vecs[hemi] = hemi_mask[:,:,0].flatten()
        all_masks.append(hemi_mask)
    all_masks = np.asarray(all_masks)
    all_masks = np.sum(all_masks, axis = 0)
    not_mask = np.logical_not(all_masks)
    hemi_matrices['inter_hemi'] = connectivity_matrices*not_mask
    hemi_mask_vecs['inter_hemi'] = not_mask[:,:,0].flatten()

    return hemi_matrices, hemi_mask_vecs


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
    hemi_matrices, hemi_mask_vecs = from_whole_brain_to_hemispheres(connectivity_matrices, atlas_index_labels)

    hemi_pca, n_components_per_network = pca_for_each_network_different_number_of_components(hemi_matrices, hemi_mask_vecs, explained_variance_th = 0.2)[0:2]
    trait_vector = load_trait_vector(trait_name, subjects)
    print(weight_by, regularization, ncm)
    linear_regression(hemi_pca, trait_vector, ['LH','RH','inter_hemi'],n_components = n_components)