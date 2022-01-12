import numpy as np
import networkx as nx


def get_efficiency(cm=None):
    """
    :param cm: matrix
    :param g: networkx graph
    enter either a graph or a matrix
    :return: network efficiency for that matrix/graph
    """

    cm = np.array(cm)
    cm = cm / np.nanmax(cm)
    cm[np.isnan(cm)] = 0
    cm = (cm / np.nansum(cm)) * 100
    cm2 = 1 / cm
    cm2[cm == 0] = 0
    g = nx.from_numpy_matrix(cm2)
    short_paths = dict(nx.all_pairs_dijkstra_path_length(g))
    d = []
    for i in short_paths.keys():
        d.extend([short_paths[i][x] for x in short_paths[i].keys() if x != i])
    eff = 1/np.array(d)
    eff[np.array(d) == 0] = 0
    eff = np.nanmean(eff)
    return eff


def get_rich_club_curve(cm, max_k):
    cm = np.array(cm)
    cm = cm / np.nanmax(cm)
    cm[np.isnan(cm)] = 0
    cm = (cm / np.nansum(cm)) * 100

    num_of_nodes = np.shape(cm)[0]
    node_degree = np.sum(cm!=0, axis=0)
    klevel = np.nanmax(node_degree)

    wrank = np.sort(cm.reshape(1,-1)[0])[::-1]
    rw = np.zeros(max_k)
    for kk in range(0,max_k):

        small_n = np.where(np.asarray(node_degree) < kk)[0]

        if len(small_n)==0:
            rw[kk] = np.nan;
            continue


        cutout_cm = cm;
        cutout_cm[small_n,:]=np.nan;
        cutout_cm[:,small_n] = np.nan;

        wr = np.nansum(cutout_cm[cutout_cm>0])

        er = np.sum(cutout_cm != 0)

        wrank_r = wrank[0:er]


        rw[kk] = wr / np.nansum(wrank_r);

    return rw
