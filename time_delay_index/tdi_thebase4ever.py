import numpy as np
import glob,os

from Time_from_node_to_node.time_mat import time_weighted_matrix, th_num_mat, n_times_random_walk
import networkx as nx

from time_delay_index.average_hcp_tdi import calc_avg_mat

main_fol = 'F:\Hila\TDI\TheBase4Ever'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}[0-9]*{os.sep}')
atlases = ['yeo7_100', 'yeo7_200', 'bnacor']
mat_type = 'time_binary'
n=500
max_steps = 50 # 100 or 300
max_path_weight = 500 # 1000 or 3000
th_types = [''] #['_highFAth', '_shortth']
for atlas in atlases:
    for cm_name_extras in th_types:
        for subj_fol in all_subj_fol:
            mat2_save = rf'{subj_fol}cm{os.sep}{atlas}_{mat_type}{cm_name_extras}_EucDist_cm_ord.npy'
            if os.path.exists(mat2_save):
                continue
            subj = subj_fol.split(os.sep)[-2]
            print(subj)
            num_mat = np.load(f'{subj_fol}cm{os.sep}{atlas}_Num{cm_name_extras}_cm_ord.npy')
            add_mat = np.load(f'{subj_fol}cm{os.sep}{atlas}_ADD{cm_name_extras}_cm_ord.npy')
            dist_mat = np.load(f'{subj_fol}cm{os.sep}{atlas}_EucDist{cm_name_extras}_cm_ord.npy')
            time_mat = time_weighted_matrix(add_mat, dist_mat)  # 1.6 is the voxel dimensions

            num_mat = th_num_mat(num_mat, 20, binarize=True)
            time_mat[num_mat == 0] = 0

            graph_num = nx.from_numpy_matrix(num_mat)
            graph_weights = nx.from_numpy_matrix(time_mat)
            time_from_node_to_node = np.zeros(num_mat.shape)
            node_list = list(graph_num.nodes())
            for start_node in node_list:
                node_vec_mean = n_times_random_walk(graph_num, graph_weights, start_node, node_list, n=n,
                                                    max_steps=max_steps, max_path_weight=max_path_weight)
                # node_vec_mean = n_times_random_walk(graph_num, graph_weights, start_node, node_list, n=500,
                #                                     max_steps=300, max_path_weight=3000)
                time_from_node_to_node[start_node, :] = node_vec_mean
            time_from_node_to_node = (time_from_node_to_node + time_from_node_to_node.T) / 2
            np.save(mat2_save, time_from_node_to_node)

        # calc_avg_mat(all_subj_fol, mat_type, main_fol + os.sep + 'cm', 'median', atlas, '')