import numpy as np
import glob,os

from Time_from_node_to_node.time_mat import time_weighted_matrix, th_num_mat, n_times_random_walk
import networkx as nx

from time_delay_index.average_hcp_tdi import calc_avg_mat

main_fol = 'F:\Hila\TDI\LeftOutAgeAnalysis'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}[0-9]*{os.sep}')
atlas = 'yeo7_100'
mat_type = 'time_th3'
n=500
max_steps = 100 # 100 or 300
max_path_weight = 1000 # 1000 or 3000
for subj_fol in all_subj_fol:
    mat2_save = rf'{subj_fol}cm{os.sep}{atlas}_{mat_type}_cm_ord.npy'
    if os.path.exists(mat2_save):
        continue
    subj = subj_fol.split(os.sep)[-2]
    print(subj)
    num_mat = np.load(f'{subj_fol}cm{os.sep}{atlas}_num_cm_ord.npy')
    add_mat = np.load(f'{subj_fol}cm{os.sep}{atlas}_add_cm_ord.npy')
    dist_mat = np.load(f'{subj_fol}cm{os.sep}{atlas}_dist_cm_ord.npy')
    time_mat = time_weighted_matrix(add_mat, dist_mat) #1.6 is the voxel dimensions

    num_mat = th_num_mat(num_mat, 3)
    time_mat[num_mat == 0] = 0

    graph_num = nx.from_numpy_matrix(num_mat)
    graph_weights = nx.from_numpy_matrix(time_mat)
    time_from_node_to_node = np.zeros(num_mat.shape)
    node_list = list(graph_num.nodes())
    for start_node in node_list:
        node_vec_mean = n_times_random_walk(graph_num, graph_weights, start_node, node_list, n=500, max_steps=300, max_path_weight=3000)
        time_from_node_to_node[start_node, :] = node_vec_mean
    np.save(mat2_save, time_from_node_to_node)


#calc_avg_mat(all_subj_fol, mat_type, main_fol + os.sep + 'cm', 'median', atlas, '')