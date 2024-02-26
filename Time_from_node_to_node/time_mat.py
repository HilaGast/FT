import numpy as np
from Time_from_node_to_node.random_walk import random_walk
import glob, os
import networkx as nx


def time_weighted_matrix(add_mat, dist_mat):
    time_mat = dist_mat / add_mat

    return time_mat


def th_num_mat(num_mat, th, binarize=False):
    """keeps only edges with strength above a certain threshold (th) and binarize the matrix if binarize=True.
    If binarize=False, the matrix is normalized between 0 to 1.
    Input:
    num_mat - the connectivity matrix
    th - the threshold
    binarize - if True, the matrix is binarized
    Output:
    num_mat - the thresholded matrix
    """
    num_th = np.nanpercentile(num_mat[num_mat > 0], th)
    if binarize:
        num_mat[num_mat < num_th] = 0
        num_mat[num_mat > 0] = 1
    else:
        num_mat[num_mat < num_th] = 0
        # Normalize between 0 to 1:
        num_mat = num_mat / np.nanmax(num_mat)

    return num_mat


def th_num_mat_sparsity(num_mat, dist_mat, sparsity=75, binarize=False):
    """keeps edges of MST (minimum spanning tree) and % of top strength edges (x is define as the sparsity parameter: x=100-sparsity)"""
    # normalize num_mat according to distance:
    num_mat[num_mat < 50] = 0
    norm_num_mat = num_mat / dist_mat
    norm_num_mat[norm_num_mat == np.inf] = 0
    norm_num_mat[norm_num_mat == -np.inf] = 0
    norm_num_mat[np.isnan(norm_num_mat)] = 0
    # keep only edges of MST:
    mst = minimum_spanning_tree(norm_num_mat)
    mask_mst = mst > 0
    # keep only top % of edges:
    th = np.nanpercentile(norm_num_mat, sparsity)
    mask_sparsity = norm_num_mat > th
    # combine masks:
    mask = mask_mst + mask_sparsity
    if binarize:
        return mask
    else:
        num_mat = num_mat * mask
        # Normalize between 0 to 1:
        num_mat = num_mat / np.nanmax(num_mat)
        return num_mat


def minimum_spanning_tree(mat):
    g = nx.from_numpy_matrix(mat)
    mst = nx.minimum_spanning_tree(g)
    mst = nx.to_numpy_array(mst)
    return mst


def node_dict_to_vec(node_dict, node_list):
    node_vec = [np.nanmean(node_dict[node]) for node in node_list]

    return node_vec


def n_times_random_walk(
    graph_num,
    graph_weights,
    start_node,
    node_list,
    n=50,
    max_steps=50,
    max_path_weight=1000,
):
    all_node_vec = np.zeros((n, len(node_list)))
    for i in range(n):
        node_dict_i = random_walk(
            graph_num, graph_weights, start_node, max_steps, max_path_weight
        )
        node_vec = node_dict_to_vec(node_dict_i, node_list)
        all_node_vec[i, :] = node_vec

    node_vec_mean = np.nanmean(all_node_vec, axis=0)

    return node_vec_mean


# if __name__ == "__main__":
# main_fol = "F:\Hila\TDI\siemens"
# experiments = ["D60d11", "D45d13", "D31d18"]
# atlases = ["yeo7_100"]
# dist_method = "DistSampAvg"
# for atlas in atlases:
#     for exp in experiments:
#         print(f"{atlas} -   {exp}")
#         all_subj_fol = glob.glob(f"{main_fol}{os.sep}{exp}{os.sep}[C,T]*{os.sep}")
#         for subj_fol in all_subj_fol:
#             file_name = rf"{subj_fol}cm{os.sep}TDI_{dist_method}_{atlas}_cm_ord.npy"
#             if os.path.exists(file_name):
#                 continue
#             subj = subj_fol.split(os.sep)[-2]
#             print(subj)
#             try:
#                 num_mat = np.load(f"{subj_fol}cm{os.sep}num_{atlas}_cm_ord.npy")
#             except FileNotFoundError:
#                 print(f"couldn't find num_mat for {subj}")
#                 continue
#             add_mat = np.load(f"{subj_fol}cm{os.sep}add_{atlas}_cm_ord.npy")
#             dist_mat = np.load(
#                 f"{subj_fol}cm{os.sep}{dist_method}_{atlas}_cm_ord.npy"
#             )
#             dist_sum = np.nansum(dist_mat)
#             if dist_sum < 100000:
#                 print(f"dist_sum is {dist_sum}, skipping {subj}")
#                 continue
#             time_mat = time_weighted_matrix(
#                 add_mat, dist_mat
#             )  # 1.25 is the voxel dimensions
#             num_mat = th_num_mat_sparsity(num_mat, dist_mat, 75, False)
#             time_mat[num_mat == 0] = 0
#
#             graph_num = nx.from_numpy_matrix(num_mat)
#             graph_weights = nx.from_numpy_matrix(time_mat)
#             time_from_node_to_node = np.zeros(num_mat.shape)
#             node_list = list(graph_num.nodes())
#             for start_node in node_list:
#                 node_vec_mean = n_times_random_walk(
#                     graph_num,
#                     graph_weights,
#                     start_node,
#                     node_list,
#                     n=2000,
#                     max_steps=int(0.25 * len(num_mat)),
#                     max_path_weight=3000,
#                 )
#                 time_from_node_to_node[start_node, :] = node_vec_mean
#             # make it symmetric:
#             time_from_node_to_node = (
#                 time_from_node_to_node + time_from_node_to_node.T
#             ) / 2
#             np.save(file_name, time_from_node_to_node)
