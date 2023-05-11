import numpy as np
import networkx as nx

from network_analysis.norm_cm_by_random_mat import rand_mat


def get_efficiency(cm=None):
    """
    :param cm: matrix
    :param g: networkx graph
    enter either a graph or a matrix
    :return: network efficiency for that matrix/graph
    """
    d = get_mean_shortest_path(cm)
    eff = 1/np.array(d)
    eff[np.array(d) == 0] = np.nan
    eff = np.nanmean(eff)
    #eff = (1/(cm.shape[0]*(cm.shape[0]-1))) * np.nansum(eff)
    return eff


def get_rich_club_curve(cm, max_k):
    cm = np.array(cm)
    cm = cm / np.nanmax(cm)
    cm[np.isnan(cm)] = 0
    #cm = (cm / np.nansum(cm)) * 100

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


def get_mean_shortest_path(cm):
    cm = np.array(cm)
    cm = 1/cm
    cm[np.isnan(cm)] = 0
    g = nx.from_numpy_matrix(cm)
    short_paths = dict(nx.all_pairs_dijkstra_path_length(g, weight='weight')) # Distances are calculated as sums of weighted edges traversed.
    d = []
    for i in short_paths.keys():
        d.extend([short_paths[i][x] for x in short_paths[i].keys() if x != i])
    d = np.asarray(d)
    d[np.isinf(d)] = np.nan
    return d


def get_clustering_coefficient(cm):
    cm = np.array(cm)
    cm[np.isnan(cm)] = 0
    g = nx.from_numpy_matrix(cm)
    cc = nx.average_clustering(g, weight='weight')

    return cc

def get_swi(cm):
    g0 = nx.from_numpy_matrix(cm)
    grand = nx.random_reference(g0, niter=1, seed=42)
    cm_rand = nx.to_numpy_matrix(grand)
    cm_rand = rand_mat(cm, 'links_shuffle')
    glattice = nx.lattice_reference(g0, niter=5, seed=42)
    cm_lattice = nx.to_numpy_matrix(glattice)

    L = np.nanmean(get_mean_shortest_path(cm))
    Lr = np.nanmean(get_mean_shortest_path(cm_rand))
    Ll = np.nanmean(get_mean_shortest_path(cm_lattice))

    C = get_clustering_coefficient(cm)
    Cr = get_clustering_coefficient(cm_rand)
    Cl = get_clustering_coefficient(cm_lattice)

    #swi = ((L-Ll)/(Lr-Ll)) * ((C-Cr)/(Cl-Cr)) #swi
    #swi = (C/Cr)/(L/Lr) #sigma
    swi = calc_swp(C,Cl,Cr,L,Ll,Lr) #swp


    #if swi < 0:
    #    swi = 0

    return swi



def calc_swp(c,cl,cr,l,ll,lr):
    # 0.4<=swp<=1 - Small world propensity
    deltaC = (cl-c)/(cl-cr)
    deltaL = (l-lr)/(ll-lr)

    if deltaC < 0:
        deltaC = 0
    if deltaL < 0:
        deltaL = 0
    if deltaC > 1:
        deltaC = 1
    if deltaL > 1:
        deltaL = 1
    swp = 1 - np.sqrt(((deltaC)**2 + (deltaL)**2)/2) #swp

    return swp
