import numpy as np
import networkx as nx

from Time_from_node_to_node.time_mat import (
    n_times_random_walk,
    time_weighted_matrix,
    th_num_mat_sparsity,
    th_num_mat,
)

TDI_FILE_NAME = ""  # resulted matrix full path file name to be saved
NUM_MAT_FILE_NAME = (
    ""  # full path of number of streamlines weighted connectivity matrix
)
ADD_MAT_FILE_NAME = ""  # full path of Axon Diameter weighted connectivity matrix
DIST_MAT_FILE_NAME = ""  # full path of distance weighted connectivity matrix
SPARSITY_TH = True  # Boolean. True if sparsity threshold is used, False otherwise
N = 2000  # number of random walks
MAX_STEPS = 25  # maximum number of steps in a random walk
MAX_PATH_WEIGHT = 10000  # maximum path weight in a random walk

num_mat = np.load(NUM_MAT_FILE_NAME)
add_mat = np.load(ADD_MAT_FILE_NAME)
dist_mat = np.load(DIST_MAT_FILE_NAME)
time_mat = time_weighted_matrix(add_mat, dist_mat)
if SPARSITY_TH:
    num_mat_th = th_num_mat_sparsity(num_mat, dist_mat, sparsity=75, binarize=False)
else:
    num_mat_th = th_num_mat(num_mat, th=50, binarize=False)
time_mat[num_mat_th == 0] = 0

# Convert matrices to graphs:
graph_num = nx.from_numpy_matrix(num_mat_th)
graph_weights = nx.from_numpy_matrix(time_mat)

# Calculate the TDI matrix N times:
TDI = np.zeros(num_mat.shape)
node_list = list(graph_num.nodes())
for start_node in node_list:
    node_vec_mean = n_times_random_walk(
        graph_num,
        graph_weights,
        start_node,
        node_list,
        n=N,
        max_steps=MAX_STEPS,
        max_path_weight=MAX_PATH_WEIGHT,
    )
    TDI[start_node, :] = node_vec_mean

# Symmetrize the matrix using average of the matrix and its transpose:
TDI = (TDI + TDI.T) / 2

# Save:
np.save(TDI_FILE_NAME, TDI)
