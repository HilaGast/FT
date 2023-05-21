import numpy as np
import glob,os

from Time_from_node_to_node.time_mat import time_weighted_matrix, th_num_mat, n_times_random_walk
import networkx as nx

from ms_h.average_time_mat_by_group import average_time_mat

main_fol = 'F:\Hila\TDI\TheBase4Ever'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}')
atlas = 'bnacor'
mat_type = 'time_th3'
for subj_fol in all_subj_fol:
    subj = subj_fol.split(os.sep)[-2]
    print(subj)
    num_mat = np.load(f'{subj_fol}cm{os.sep}num_{atlas}_cm_ord.npy')
    add_mat = np.load(f'{subj_fol}cm{os.sep}add_{atlas}_cm_ord.npy')
    dist_mat = np.load(f'{subj_fol}cm{os.sep}dist_{atlas}_cm_ord.npy')
    time_mat = time_weighted_matrix(add_mat, dist_mat/1.6) #1.6 is the voxel dimensions

    num_mat = th_num_mat(num_mat, 3)
    time_mat[num_mat == 0] = 0

    graph_num = nx.from_numpy_matrix(num_mat)
    graph_weights = nx.from_numpy_matrix(time_mat)
    time_from_node_to_node = np.zeros(num_mat.shape)
    node_list = list(graph_num.nodes())
    for start_node in node_list:
        node_vec_mean = n_times_random_walk(graph_num, graph_weights, start_node, node_list, n=100)
        time_from_node_to_node[start_node, :] = node_vec_mean
    np.save(rf'{subj_fol}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy', time_from_node_to_node)

average_time_mat(all_subj_fol,'TheBase4Ever',main_fol,mat_type,atlas)