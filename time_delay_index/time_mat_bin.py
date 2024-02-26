import glob, os
import numpy as np
import networkx as nx
from Time_from_node_to_node.time_mat import time_weighted_matrix, th_num_mat, n_times_random_walk


def bin_time_mat_siemens():
    main_fol = 'F:\Hila\TDI\siemens'
    exp = 'D31d18'
    all_subj_fol = glob.glob(f'{main_fol}{os.sep}{exp}{os.sep}[C,T]*{os.sep}')
    atlases = ['yeo7_100']

    for atlas in atlases:
        print(f'{atlas} -   {exp}')
        for subj_fol in all_subj_fol:
            file_name = rf'{subj_fol}cm{os.sep}TDIbin_{atlas}_cm_ord.npy'
            if os.path.exists(file_name):
                continue
            subj = subj_fol.split(os.sep)[-2]
            print(subj)
            try:
                num_mat = np.load(f'{subj_fol}cm{os.sep}num_{atlas}_cm_ord.npy')
            except FileNotFoundError:
                print(f"couldn't find num_mat for {subj}")
                continue
            add_mat = np.load(f'{subj_fol}cm{os.sep}add_{atlas}_cm_ord.npy')
            dist_mat = np.load(f'{subj_fol}cm{os.sep}EucDist_{atlas}_cm_ord.npy')
            time_mat = time_weighted_matrix(add_mat, dist_mat)  # 1.25 is the voxel dimensions

            num_mat = th_num_mat(num_mat, 75, binarize=True)
            time_mat[num_mat == 0] = 0

            graph_num = nx.from_numpy_matrix(num_mat)
            graph_weights = nx.from_numpy_matrix(time_mat)
            time_from_node_to_node = np.zeros(num_mat.shape)
            node_list = list(graph_num.nodes())
            for start_node in node_list:
                node_vec_mean = n_times_random_walk(graph_num, graph_weights, start_node, node_list, n=500,
                                                    max_steps=int(0.25*len(num_mat)), max_path_weight=3000)
                time_from_node_to_node[start_node, :] = node_vec_mean
            # make it symmetric:
            time_from_node_to_node = (time_from_node_to_node + time_from_node_to_node.T) / 2
            np.save(file_name, time_from_node_to_node)


if __name__ == '__main__':
    bin_time_mat_siemens()
