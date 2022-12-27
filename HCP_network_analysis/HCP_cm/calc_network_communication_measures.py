import numpy as np
import networkx as nx
from network_analysis.topology_rep import find_largest_connected_component


def calc_spe(cm):
    cm = np.array(cm)
    cm[np.isnan(cm)] = 0
    cm[cm<0] = 0
    g = nx.from_numpy_matrix(cm)
    g = find_largest_connected_component(g, show=False)
    short_paths = dict(nx.all_pairs_dijkstra_path_length(g))
    spe = np.zeros(cm.shape)

    for i in short_paths.keys():
        for x in short_paths[i].keys():
            if x != i:
                spe[i,x] = 1/short_paths[i][x]

    spe = (spe+spe.T)/2
    return spe


def calc_ne(cm, cm_dist):
    cm = np.array(cm)
    cm[np.isnan(cm)] = 0
    cm[cm<0] = 0
    g = nx.from_numpy_matrix(cm)
    g = find_largest_connected_component(g, show=False)
    inf_eye = np.eye(cm.shape[0], cm.shape[1]) * np.inf
    inf_eye[np.isnan(inf_eye)] = 1
    nav = np.zeros(cm.shape)
    for i in g.nodes:
        for j in g.nodes:
            if i != j:
                curr_node = i
                last_node = curr_node
                target = j

                pl_wei = 0
                nodes_path = []
                while curr_node != target:

                    neighbors = list(g.neighbors(curr_node))
                    dist_from_neighbor = [cm_dist[target,nei] for nei in neighbors]
                    min_index = dist_from_neighbor.index(min(dist_from_neighbor)) # returns only the first
                    next_node = neighbors[min_index]
                    if not next_node or next_node == last_node:
                        pl_wei = np.inf
                        continue
                    if next_node in nodes_path:
                        pl_wei = np.inf
                        print('loop')
                        continue

                    pl_wei += cm[curr_node, next_node]
                    nodes_path.append(next_node)
                    last_node = curr_node
                    curr_node = next_node

                nav[i, j] = pl_wei
        nav = nav * inf_eye
        ne = 1/nav
        ne = (ne + ne.T)/2
    return ne


def calc_cmy(cm):
    cm = np.array(cm)
    cm[np.isnan(cm)] = 0
    g = nx.from_numpy_matrix(cm)
    cmy = np.zeros(cm.shape)
    cmy_dict = nx. communicability_exp(g)
    for i in cmy_dict.keys():
        for j in cmy_dict[i].keys():
            cmy[i,j] = int(cmy_dict[i][j])

    cmy = (cmy+cmy.T)/2

    return cmy
