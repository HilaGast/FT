import numpy as np

def from_whole_brain_to_networks(connectivity_matrices, atlas_index_labels, hemi_flag=True):

    labels_file = open(atlas_index_labels, 'r', errors='ignore')
    labels_name = labels_file.readlines()
    labels_file.close()
    labels_networks = find_network_names(labels_name, hemi_flag)
    network_mask_dict = create_dict_of_networks_indices(labels_name, labels_networks, hemi_flag)
    if hemi_flag:
        networks_matrices, network_mask_vecs = create_networks_hemi_matrices(connectivity_matrices, network_mask_dict)
    else:
        networks_matrices, network_mask_vecs = create_networks_matrices(connectivity_matrices, network_mask_dict)

    return networks_matrices, network_mask_vecs

def find_network_names(labels_name, hemi_flag):
    label_networks = []
    for l in labels_name:
        label_parts = l.split('\t')
        if hemi_flag:
            label_networks.append(label_parts[1].split('_')[2]+'_'+label_parts[1].split('_')[1])
        else:
            label_networks.append(label_parts[1].split('_')[2])
    label_networks = list(set(label_networks))

    return label_networks

def create_networks_hemi_matrices(connectivity_matrices, network_mask_dict):
    networks_matrices = {}
    network_mask_vecs = {}
    inter_network_mask_lh = np.zeros(connectivity_matrices.shape, dtype = bool)
    inter_network_mask_rh = np.zeros(connectivity_matrices.shape, dtype=bool)
    for network1 in network_mask_dict.keys():
        network1_name_parts = network1.split('_')
        for network2 in network_mask_dict.keys():
            network2_name_parts = network2.split('_')
            network_mask = np.zeros(connectivity_matrices.shape, dtype = bool)

            for r in network_mask_dict[network1]:
                for c in network_mask_dict[network2]:
                    network_mask[r, c, :] = True
            if network1_name_parts[0]==network2_name_parts[0] and network1_name_parts[1]==network2_name_parts[1]:
                networks_matrices[network1] = connectivity_matrices*network_mask
                network_mask_vecs[network1] = network_mask[:,:,0].flatten()
            elif network1_name_parts[0]==network2_name_parts[0] and network1_name_parts[1]!=network2_name_parts[1]:
                networks_matrices[network1_name_parts[0]] = connectivity_matrices*network_mask
                network_mask_vecs[network1_name_parts[0]] = network_mask[:,:,0].flatten()
            elif network1_name_parts[1]==network2_name_parts[1]:
                if network1_name_parts[1] == 'LH':
                    inter_network_mask_lh+=network_mask
                elif network2_name_parts[1] == 'RH':
                    inter_network_mask_rh += network_mask

    networks_matrices['inter_network_LH'] = connectivity_matrices*inter_network_mask_lh
    network_mask_vecs['inter_network_LH'] = inter_network_mask_lh[:,:,0].flatten()
    networks_matrices['inter_network_RH'] = connectivity_matrices*inter_network_mask_rh
    network_mask_vecs['inter_network_RH'] = inter_network_mask_rh[:,:,0].flatten()

    return networks_matrices, network_mask_vecs

def create_networks_matrices(connectivity_matrices, network_mask_dict):
    networks_matrices = {}
    all_masks = []
    network_mask_vecs = {}

    for network in network_mask_dict.keys():
        network_mask = np.zeros(connectivity_matrices.shape, dtype = bool)
        for r in network_mask_dict[network]:
            for c in network_mask_dict[network]:
                network_mask[r, c, :] = True
        networks_matrices[network] = connectivity_matrices*network_mask
        network_mask_vecs[network] = network_mask[:,:,0].flatten()
        all_masks.append(network_mask)
    all_masks = np.asarray(all_masks)
    all_masks = np.sum(all_masks, axis = 0)
    not_mask = np.logical_not(all_masks)
    networks_matrices['inter_network'] = connectivity_matrices*not_mask
    network_mask_vecs['inter_network'] = not_mask[:,:,0].flatten()

    return networks_matrices, network_mask_vecs

def create_dict_of_networks_indices(labels_name, label_networks, hemi_flag):
    network_mask_dict = {}
    for network in label_networks:
        network_mask_dict[network] = []
    for l in labels_name:
        label_parts = l.split('\t')
        if hemi_flag:
            network_mask_dict[label_parts[1].split('_')[2]+'_'+label_parts[1].split('_')[1]].append(int(label_parts[0]) - 1)
        else:
            network_mask_dict[label_parts[1].split('_')[2]].append(int(label_parts[0])-1)

    return network_mask_dict