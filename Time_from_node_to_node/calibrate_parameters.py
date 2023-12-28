import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from Time_from_node_to_node.time_mat import (
    time_weighted_matrix,
    th_num_mat_sparsity,
    n_times_random_walk,
)


def choose_n_random_walk_repetitions(atlas, exp, subj, subj_i, n_vec, path):
    subj_fol = rf"F:\Hila\TDI\siemens\{exp}\{subj}"
    num_mat = np.load(f"{subj_fol}{os.sep}cm{os.sep}num_{atlas}_cm_ord.npy")
    add_mat = np.load(f"{subj_fol}{os.sep}cm{os.sep}add_{atlas}_cm_ord.npy")
    euc_mat = np.load(f"{subj_fol}{os.sep}cm{os.sep}DistMode_{atlas}_cm_ord.npy")
    time_mat = time_weighted_matrix(add_mat, euc_mat)
    num_mat = th_num_mat_sparsity(num_mat, euc_mat, 75, False)
    time_mat[num_mat == 0] = 0

    n_areas = num_mat.shape[0]
    steps = n_areas
    std_ni = np.zeros((len(n_vec), n_areas, n_areas))
    i_ni = 0
    for ni in n_vec:
        mat_n = []
        for i in range(5):
            graph_num = nx.from_numpy_matrix(num_mat)
            graph_weights = nx.from_numpy_matrix(time_mat)
            time_from_node_to_node = np.zeros(num_mat.shape)
            node_list = list(graph_num.nodes())
            for start_node in node_list:
                node_vec_mean = n_times_random_walk(
                    graph_num,
                    graph_weights,
                    start_node,
                    node_list,
                    n=ni,
                    max_steps=steps,
                    max_path_weight=path,
                )
                time_from_node_to_node[start_node, :] = node_vec_mean
            # make it symmetric:
            time_from_node_to_node = (
                time_from_node_to_node + time_from_node_to_node.T
            ) / 2
            mat_n.append(time_from_node_to_node)
            vec = time_from_node_to_node.flatten()
            vec[np.isnan(vec)] = 0
        std = np.nanstd(mat_n, axis=0)
        std_ni[i_ni, :, :] = std
        i_ni += 1
    std_ni = std_ni.reshape((len(n_vec), -1)).T
    sns.boxplot(std_ni, color="red", width=0.3)
    plt.xticks(np.arange(len(n_vec)), n_vec)
    plt.xlabel("n repetitions of random walk")
    plt.ylabel("std for 5 repetitions of n random walks")
    plt.title(f"{atlas} - subj {subj_i}")
    plt.show()


def choose_max_steps_random_walk(atlas, exp, subj, subji, steps_ratio_vec, path, n):
    subj_fol = rf"F:\Hila\TDI\siemens\{exp}\{subj}"
    num_mat = np.load(f"{subj_fol}{os.sep}cm{os.sep}num_{atlas}_cm_ord.npy")
    add_mat = np.load(f"{subj_fol}{os.sep}cm{os.sep}add_{atlas}_cm_ord.npy")
    euc_mat = np.load(f"{subj_fol}{os.sep}cm{os.sep}DistMode_{atlas}_cm_ord.npy")
    time_mat = time_weighted_matrix(add_mat, euc_mat)
    num_mat = th_num_mat_sparsity(num_mat, euc_mat, 75, False)
    time_mat[num_mat == 0] = 0

    n_areas = num_mat.shape[0]
    steps = [int(n_areas * steps_ratio) for steps_ratio in steps_ratio_vec]
    sparsity_vec = np.zeros(len(steps_ratio_vec))
    for i, step in enumerate(steps):
        graph_num = nx.from_numpy_matrix(num_mat)
        graph_weights = nx.from_numpy_matrix(time_mat)
        time_from_node_to_node = np.zeros(num_mat.shape) * np.nan
        node_list = list(graph_num.nodes())
        for start_node in node_list:
            node_vec_mean = n_times_random_walk(
                graph_num,
                graph_weights,
                start_node,
                node_list,
                n=n,
                max_steps=step,
                max_path_weight=path,
            )
            time_from_node_to_node[start_node, :] = node_vec_mean
        # make it symmetric:
        time_from_node_to_node = (time_from_node_to_node + time_from_node_to_node.T) / 2
        time_from_node_to_node[np.isnan(time_from_node_to_node)] = 0
        sparsity = np.sum(time_from_node_to_node == 0) / (n_areas**2)
        sparsity_vec[i] = sparsity

    plt.scatter(np.asarray(steps_ratio_vec) * 100, sparsity_vec * 100)
    plt.xlabel("% from number of nodes in the atlas")
    plt.ylabel("TDI matrix sparsity")
    plt.ylim(0, 2.5)
    plt.title(f"{atlas} - subj {subji}")
    plt.show()


if __name__ == "__main__":
    atlas = "yeo7_100"
    subj_list = [
        "C1156_01",
        "C1156_02",
        "C1156_04",
        "C1156_05",
        "C1156_06",
        "C1156_07",
        "T0156_01",
        "T0156_02",
        "T1156_11",
    ]
    n_vec = [10, 100, 500, 1000, 2000]
    step_ratio_vec = [0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 0.9, 1]
    path = 3000
    exp = "D31d18"
    for si, subj in enumerate(subj_list):
        # choose_n_random_walk_repetitions(atlas, exp, subj, si, n_vec, path)
        choose_max_steps_random_walk(atlas, exp, subj, si, step_ratio_vec, path, 1000)
