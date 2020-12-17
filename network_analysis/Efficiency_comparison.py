from all_subj import *
import numpy as np
from weighted_tracts import nodes_labels_yeo7
import networkx as nx
from network_analysis.topology_rep import find_largest_connected_component


def from_mat_2_graph(mat,id,return_connected = True):
    mat_weights = mat[id]
    mat_weights = mat_weights[:, id]
    mat = np.zeros((mat_weights.shape[0] + 1, mat_weights.shape[1] + 1))
    mat[1:, 1:] = mat_weights
    g = nx.from_numpy_array(mat)
    #print(f'*** \n Graph has {len(g.nodes)} nodes and {len(g.edges)} edges \n***')

    if return_connected:
        g = find_largest_connected_component(g, show=False)
        print(f'*** \n Connected component has {len(g.nodes)} nodes and {len(g.edges)} edges \n***')

    return g


if __name__== '__main__':
    aspl = np.zeros((3,len(all_subj_names)))
    eff = np.zeros((3,len(all_subj_names)))
    i=0
    for s, n in zip(all_subj_folders[::], all_subj_names[::]):
        mat_file_ax = subj_folder+ s+ r'\weighted_wholebrain_4d_labmask_yeo7_200_nonnorm.npy'
        mat_ax = np.load(mat_file_ax)
        mat_norm_ax = 1 / mat_ax * 8.75
        mat_norm_ax[np.isinf(mat_norm_ax)] = 0
        mat_norm_ax[np.isnan(mat_norm_ax)] = 0

        mat_file_num = subj_folder+ s+ r'\non-weighted_wholebrain_4d_labmask_yeo7_200_nonnorm.npy'
        mat_num = np.load(mat_file_num)
        mat_norm_num = 1 / mat_num
        mat_norm_num[np.isinf(mat_norm_num)] = 0
        mat_norm_num[np.isnan(mat_norm_num)] = 0

        idx = nodes_labels_yeo7(index_to_text_file)[1]
        id = np.argsort(idx)

        g_ax = from_mat_2_graph(mat_norm_ax, id)
        g_num = from_mat_2_graph(mat_norm_num, id)

        aspl_ax = nx.average_shortest_path_length(g_ax,weight='weight',method='dijkstra')
        aspl_num = nx.average_shortest_path_length(g_num,weight='weight',method='dijkstra')
        aspl_nw = nx.average_shortest_path_length(g_num,weight=None)
        aspl[:,i] = [aspl_nw,aspl_num,aspl_ax]

        eff_ax = 1/aspl_ax
        eff_num = 1/aspl_num
        eff_nw = 1/aspl_nw
        eff[:,i] = [eff_nw,eff_num,eff_ax]

        i+=1









