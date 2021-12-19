
import numpy as np
import networkx as nx


def merge_dict(dict1, dict2):
   ''' Merge dictionaries and keep values of common keys in list'''
   import more_itertools
   dict3 = {**dict1, **dict2}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = [value , dict1[key]]

   for key, value in dict3.items():
       dict3[key] = list(more_itertools.collapse(value))  # makes sure it is a flattened list

   return dict3


def get_local_efficiency(cm):
    """
    :param cm: matrix
    :return: network efficiency for each node in the matrix.
    The local efficiency of a node is computed by the global efficiency of it's neighbors.

    """

    cm = np.array(cm)
    cm = cm / np.nanmax(cm)
    cm[np.isnan(cm)] = 0
    cm = (cm / np.nansum(cm)) * 100
    cm2 = 1 / cm
    cm2[cm == 0] = 0
    g = nx.from_numpy_matrix(cm2)
    short_paths = dict(nx.all_pairs_dijkstra_path_length(g))
    eff_dict={}
    for i in short_paths.keys():
        d = [short_paths[i][x] for x in g.neighbors(i) if x != i]
        eff = 1/np.array(d)
        eff[d==0]=0
        eff_dict[i] = np.nanmean(eff)

    return eff_dict


def get_node_degree(cm):

    cm = np.array(cm)
    cm = cm / np.nanmax(cm)
    cm[np.isnan(cm)] = 0
    cm = (cm / np.nansum(cm)) * 100
    nd = np.nansum(cm, axis=0)

    return nd


def get_node_betweenness_centrality(cm):
    cm = np.array(cm)
    cm = cm / np.nanmax(cm)
    cm[np.isnan(cm)] = 0
    cm = (cm / np.nansum(cm)) * 100
    cm2 = 1 / cm
    cm2[cm == 0] = 0
    g = nx.from_numpy_matrix(cm2)
    betweenness_centrality = nx.betweenness_centrality(g,weight='weight')
    bc = np.zeros(cm.shape[0])
    for i in betweenness_centrality.keys():
        bc[i] = betweenness_centrality[i]

    return bc