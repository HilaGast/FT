import numpy as np
from Time_from_node_to_node.random_walk import *
import glob, os

def time_weighted_matrix(add_mat, dist_mat):

    time_mat = dist_mat/ add_mat

    return time_mat


def th_num_mat(num_mat, th):

        num_mat[num_mat < th] = 0

        return num_mat

def node_dict_to_vec(node_dict, node_list):

    node_vec = [np.nanmean(node_dict[node]) for node in node_list]

    return node_vec

def n_times_random_walk(graph_num, graph_weights, start_node, node_list, n = 50, max_steps = 50, max_path_weight = 1000):

    all_node_vec = np.zeros((n, len(node_list)))
    for i in range(n):
        node_dict_i = random_walk(graph_num, graph_weights, start_node, max_steps, max_path_weight)
        node_vec = node_dict_to_vec(node_dict_i, node_list)
        all_node_vec[i, :] = node_vec

    node_vec_mean = np.nanmean(all_node_vec, axis = 0)

    return node_vec_mean

if __name__ == '__main__':
    main_fol = 'F:\Hila\TDI\siemens'
    all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}')
    exp = 'D31d18'
    atlas = 'yeo7_200'
    for subj_fol in all_subj_fol:
        if 'group' in subj_fol or 'surfaces' in subj_fol:
            continue
        file_name = rf'{subj_fol}{exp}{os.sep}cm{os.sep}time_th3_{atlas}_cm_ord.npy'
        if os.path.exists(file_name):
            continue
        subj = subj_fol.split(os.sep)[-2]
        print(subj)
        num_mat = np.load(f'{subj_fol}{exp}{os.sep}cm{os.sep}num_{atlas}_cm_ord.npy')
        add_mat = np.load(f'{subj_fol}{exp}{os.sep}cm{os.sep}add_{atlas}_cm_ord.npy')
        dist_mat = np.load(f'{subj_fol}{exp}{os.sep}cm{os.sep}dist_{atlas}_cm_ord.npy')
        time_mat = time_weighted_matrix(add_mat, dist_mat) #1.25 is the voxel dimensions

        num_mat = th_num_mat(num_mat, 3)
        time_mat[num_mat == 0] = 0

        graph_num = nx.from_numpy_matrix(num_mat)
        graph_weights = nx.from_numpy_matrix(time_mat)
        time_from_node_to_node = np.zeros(num_mat.shape)
        node_list = list(graph_num.nodes())
        for start_node in node_list:
            node_vec_mean = n_times_random_walk(graph_num, graph_weights, start_node, node_list, n=500, max_steps=100, max_path_weight=1000)
            time_from_node_to_node[start_node, :] = node_vec_mean
        np.save(file_name, time_from_node_to_node)
