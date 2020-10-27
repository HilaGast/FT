import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from weighted_tracts import *
from network_analysis.create_labels_centroid_2d import create_nodes_position
from network_analysis.topology_rep import *
import scipy.io as sio
import matplotlib.cm as cm


def load_communities(g,communities_file,weight_by):
    mat = sio.loadmat(communities_file)
    #communities = np.asarray(mat['ciuall'])
    communities = np.asarray(mat['ciuv'])
    if weight_by == 'num':
        c = communities[:,0]
    elif weight_by == 'fa':
        c = communities[:, 1]
    elif weight_by == 'ax':
        c = communities[:, 2]
    c_connected = []
    for n in g.nodes:
        g.nodes[n]['community'] = c[n-1]
        c_connected.append(c[n-1])

    return c_connected


def load_mat_2_graph(mat_file):
    mat = np.load(mat_file)
    labels,idx = nodes_labels_aal3(index_to_text_file)
    id = np.argsort(idx)
    mat_weights = mat[id]
    mat_weights = mat_weights[:, id]
    mat = np.zeros((mat_weights.shape[0]+1,mat_weights.shape[1]+1))
    mat[1:,1:]=mat_weights
    #mat=mat/np.max(mat[:])
    g = nx.from_numpy_array(mat)

    return g, labels


def set_node_label(g,labels):
    label_dict = {}
    for n,l in zip(g.nodes,labels):
        g.nodes[n]['label'] = l
        label_dict[n] = l
    return label_dict


def show_topology(mat_file,communities_file,weight_by, dup=200, is_deg_norm=True, is_edge_norm = True):
    g, labels = load_mat_2_graph(mat_file)
    print(f'*** \n Graph has {len(g.nodes)} nodes and {len(g.edges)} edges \n***')
    g = find_largest_connected_component(g, show=False)
    print(f'*** \n Connected component has {len(g.nodes)} nodes and {len(g.edges)} edges \n***')
    label_dict = set_node_label(g, labels)

    c=load_communities(g,communities_file, weight_by)
    deg = [np.sum(nx.to_numpy_array(g), 0)]
    set_edge_community(g)
    selected_nodes = list(g)
    pos = create_nodes_position(atlas='aal3')
    pos = {k: v for k, v in pos.items() if k in selected_nodes}
    cmap = cm.get_cmap('gist_rainbow', max(c) + 1)
    plt.figure(1, [50, 50])
    if is_deg_norm:
        degnorm = norm_deg(deg,dup=dup)
        nx.draw_networkx_nodes(g, pos, list(g.nodes), node_size=degnorm, cmap=cmap, node_color=c)

    else:
        nx.draw_networkx_nodes(g, pos, list(g.nodes), cmap=cmap, node_size=5000, node_color=c)

    external = [(v, w) for v, w in g.edges if g.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in g.edges if g.edges[v, w]['community'] > 0]
    internal_color = ['dimgray' for e in internal]
    edgewidth = [d['weight'] for (u, v, d) in g.edges(data=True)]

    if is_edge_norm:
        normwidth = norm_edge(edgewidth)
        nx.draw_networkx_edges(g, pos, width=normwidth, alpha=0.2, edge_color='silver')
        nx.draw_networkx_edges(g, pos, width=normwidth, alpha=0.9, edgelist=internal, edge_color=internal_color)

    else:
        nx.draw_networkx_edges(g, pos, alpha=0.2, edge_color='silver')
        nx.draw_networkx_edges(g, pos, alpha=0.9, edgelist=internal, edge_color=internal_color)

    nx.draw_networkx_labels(g, pos,label_dict, font_size=15, font_weight='bold')
    plt.show()


def norm_deg(deg, dup=200):
    degmin = np.min(deg)
    degmax = np.max(deg)
    degnorm = [(20 + dup * (d - degmin) / (degmax - degmin)) ** 2 for d in deg]
    return degnorm


def norm_edge(edgewidth):
    wmin = np.min(edgewidth)
    wmax = np.max(edgewidth)
    normwidth = np.asarray([1 + 30 * ((w - wmin) / (wmax - wmin)) ** 7 for w in edgewidth])
    return normwidth


if __name__== '__main__':
    subj = all_subj_folders
    names = all_subj_names

    for s,n in zip(subj,names):
        print(n)
        folder_name = subj_folder + s

        #weight_by = 'fa'
        #weight_by = 'num'
        weight_by = 'ax'

        num_file = 'non-weighted_mega_wholebrain_4d_labmask_aal3_nonnorm.npy'
        ax_file = 'weighted_mega_wholebrain_4d_labmask_aal3_nonnorm.npy'
        fa_file = 'weighted_mega_wholebrain_4d_labmask_aal3_FA_nonnorm.npy'


        if weight_by == 'fa':
            mat_file = rf'{folder_name}\{fa_file}'
        elif weight_by == 'num':
            mat_file = rf'{folder_name}\{num_file}'
        elif weight_by == 'ax':
            mat_file = rf'{folder_name}\{ax_file}'
        else:
            msg = 'No weight found'
            print(msg)

        for file in os.listdir(folder_name):
            if 'subj_communities' in file:
                communities_file = rf'{folder_name}\{file}'

        show_topology(mat_file, communities_file, weight_by, dup=200, is_deg_norm=True, is_edge_norm=True)




    #folder_name = r'F:\Hila\Ax3D_Pack\mean_vals\aal3_atlas\mean_' #for mean of 50 subjects
    #communities_file = r'C:\Users\HilaG\Desktop\4OlafSporns\matrices(float64)\group_division_2weights_allsubj.mat'
