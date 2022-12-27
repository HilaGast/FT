import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from HCP_network_analysis.core_periphery_representation import load_mat_2_graph
from network_analysis.create_labels_centroid_2d import create_nodes_position
from network_analysis.topology_rep import *
import scipy.io as sio
import matplotlib.cm as cm
import os
from scipy.stats import rankdata

def load_communities(g,communities_vec):

    c_connected = []
    for n in g.nodes:
        g.nodes[n]['community'] = communities_vec[n-1]
        c_connected.append(communities_vec[n-1])

    return c_connected


def set_node_label(g,labels):
    label_dict = {}
    for n,l in zip(g.nodes,labels):
        g.nodes[n]['label'] = l
        label_dict[n] = l
    return label_dict


def find_nocommunity_nodes(g):
    no_community = []
    for n in g.nodes:
        if g.nodes[n]['community'] == 0:
            no_community.append(n)
    return no_community


def show_topology(mat_file,communities_vec,weight_by,atlas, eflag, dup=200, nodes_norm_by='bc', is_edge_norm = True):
    g = load_mat_2_graph(mat_file,atlas, eflag)
    print(f'*** \n Graph has {len(g.nodes)} nodes and {len(g.edges)} edges \n***')
    g = find_largest_connected_component(g, show=False)
    print(f'*** \n Connected component has {len(g.nodes)} nodes and {len(g.edges)} edges \n***')

    c=load_communities(g,communities_vec)
    cr = rankdata(c, method='dense')
    set_edge_community(g)
    selected_nodes = list(g)
    pos = create_nodes_position(atlas=atlas)
    pos = {k: v for k, v in pos.items() if k in selected_nodes}
    print(set(cr))
    cmap = cm.get_cmap('Set1_r', max(cr) + 1)
    nc = find_nocommunity_nodes(g)
    nci = list(np.asarray(nc)-1)
    from HCP_network_analysis.laterality import comp_li
    li = comp_li(cr,nci, atlas=atlas)
    print(f'LI: {li}%')
    plt.figure(1, [50, 50])


    if nodes_norm_by == 'deg':
        '''Normalize nodes size by degree'''
        deg = [np.sum(nx.to_numpy_array(g), 0)]
        degnorm = norm_deg(deg,dup=dup)
        nx.draw_networkx_nodes(g, pos, list(g.nodes), node_size=degnorm, cmap=cmap, node_color=cr)
        nx.draw_networkx_nodes(g, pos, nodelist=nc, node_size=degnorm[0][nci], node_color='silver')
    elif nodes_norm_by == 'bc':
        '''Normalize nodes size by betweeness centrality'''
        bc = nx.betweenness_centrality(g,weight='weight')
        nx.draw_networkx_nodes(g, pos, list(g.nodes), node_size=np.asarray(list(bc.values()))*1e6, cmap=cmap, node_color=cr)
        nx.draw_networkx_nodes(g, pos, nodelist=nc, node_size=np.asarray(list(bc.values()))*1e6[nci], node_color='silver')
    else:
        nx.draw_networkx_nodes(g, pos, list(g.nodes), cmap=cmap, node_size=5000, node_color=cr)
        nx.draw_networkx_nodes(g, pos, nodelist=nc, node_color='silver')

    external = [(v, w) for v, w in g.edges if g.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in g.edges if g.edges[v, w]['community'] > 0]
    internal_color = ['dimgray' for e in internal]
    edgewidth = [d['weight'] for (u, v, d) in g.edges(data=True)]

    if is_edge_norm:
        normwidth = norm_edge(edgewidth)
        nx.draw_networkx_edges(g, pos, width=normwidth*0.3, alpha=0.2, edge_color='silver')
        nx.draw_networkx_edges(g, pos, width=normwidth, alpha=0.9, edgelist=internal, edge_color=internal_color)

    else:
        nx.draw_networkx_edges(g, pos, alpha=0.2, edge_color='silver')
        nx.draw_networkx_edges(g, pos, alpha=0.9, edgelist=internal, edge_color=internal_color)

    nx.draw_networkx_labels(g, pos, font_size=15, font_weight='bold')
    plt.title(f'{weight_by} \n LI: {round(li)}%',fontsize=80)
    plt.savefig(f'G:\data\V7\HCP\communities\Figs\{weight_by}.png')
    plt.show()



def norm_deg(deg, dup=200):
    degmin = np.min(deg)
    degmax = np.max(deg)
    degnorm = [(20 + dup * (d - degmin) / (degmax - degmin)) ** 2 for d in deg]
    return degnorm


def norm_edge(edgewidth):
    wmin = np.nanmin(edgewidth)
    wmax = np.nanmax(edgewidth)
    normwidth = np.asarray([15 * ((w - wmin) / (wmax - wmin)) for w in edgewidth])
    normmean = np.nanmean(normwidth)
    normstd = np.nanstd(normwidth)
    normwidth[normwidth<[normmean+normstd]] = np.nan
    return normwidth


def run_community_top_by_subj(weight_by,nodes_norm = 'bc'):
    subj = all_subj_folders
    names = all_subj_names
    for s,n in zip(subj,names):
        print(n)
        folder_name = subj_folder + s


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

        show_topology(mat_file, communities_file, weight_by, dup=200, nodes_norm_by=nodes_norm, is_edge_norm=True)


def run_community_top_by_group(weight_by, atlas, bin_flag, nodes_norm='deg',th='', exp_flag=True):
    folder_name = r'G:\data\V7\HCP\cm'
    num_file = f'{atlas}_average_num{th}.npy'
    ax_file = f'{atlas}_average_add{th}.npy'
    fa_file = f'{atlas}_average_fa{th}.npy'
    eflag = exp_flag
    if weight_by == 'fa':
        mat_file = rf'{folder_name}\{fa_file}'
        if exp_flag:
            eflag=True
    elif weight_by == 'num':
        mat_file = rf'{folder_name}\{num_file}'
    elif weight_by == 'ax':
        mat_file = rf'{folder_name}\{ax_file}'
        if exp_flag:
            eflag=True
    else:
        msg = 'No weight found'
        print(msg)

    for file in os.listdir(folder_name):
        if f'group_division_{atlas}{th}{bin_flag}.mat' in file:
            communities_file = rf'{folder_name}\{file}'

    mat = sio.loadmat(communities_file)
    #communities = np.asarray(mat['ciuall'])
    communities = np.asarray(mat['ciuv'])
    if weight_by == 'num':
        c = communities[:,0]
    elif weight_by == 'fa':
        c = communities[:, 1]
    elif weight_by == 'ax':
        c = communities[:, 2]

    show_topology(mat_file, c, weight_by, eflag, dup=100, nodes_norm_by=nodes_norm, is_edge_norm=True)


if __name__== '__main__':


    weight_by = 'fa'
    #weight_by = 'num'
    #weight_by = 'ax'
    atlas = 'yeo7_200'
    bin_flag=''
    run_community_top_by_group(weight_by, atlas, bin_flag, th='_histmatch_th', exp_flag=False)

