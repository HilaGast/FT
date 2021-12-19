import networkx as nx
import numpy as np
import os

def mat2graph(cm):
    g = nx.from_numpy_array(cm)

    return g

def calc_ebc(g):

    ebc = nx.edge_betweenness_centrality(g, weight='weight', normalized=True)

    return ebc

def mat_ebc(cm,ebc=None):
    if not ebc:
        g = mat2graph(cm)
        ebc = calc_ebc(g)
    mat_ebc = np.zeros(cm.shape)
    for (u,v) in ebc.keys():
        mat_ebc[u,v] = ebc[(u,v)]
        mat_ebc[v,u] = ebc[(u,v)]

    return mat_ebc

def save_ebc_mat(mat_ebc,folder_name, weighted_type):
    np.save(mat_ebc,f'{folder_name}{os.sep}EBC_{weighted_type}.npy')