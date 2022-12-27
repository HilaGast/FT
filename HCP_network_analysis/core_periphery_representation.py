import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from network_analysis.create_labels_centroid_2d import create_nodes_position
from network_analysis.topology_rep import find_largest_connected_component, set_edge_community
import scipy.io as sio
import matplotlib.cm as cm
import os, glob
from Tractography.find_atlas_labels import *


def load_core_periphery(g, core_periphery_vec):

    c_connected = []
    core_n = []
    for n in g.nodes:
        g.nodes[n]['community'] = core_periphery_vec[n-1]
        c_connected.append(core_periphery_vec[n-1])
        if core_periphery_vec[n-1]>0:
            core_n.append(n)
    lh = idx_2_label(r'G:\data\atlases\yeo\yeo7_200\index2label.txt', core_n, 'yeo7_200',True)

    return c_connected


def load_mat_2_graph(mat_file,atlas, eflag= False):
    mat = np.load(mat_file)
    if eflag:
        mat = np.exp(mat)
    main_folder = os.path.dirname(mat_file)
    idx = np.load(rf'{main_folder}\{atlas}_cm_ord_lookup.npy')
    id = np.argsort(idx)
    mat_weights = mat[id]
    mat_weights = mat_weights[:, id]
    mat = np.zeros((mat_weights.shape[0]+1,mat_weights.shape[1]+1))
    mat[1:,1:]=mat_weights
    g = nx.from_numpy_array(mat)

    return g


def show_topology(mat_file,core_periphery_vec,weight_by, dup=200, atlas='bna', nodes_norm_by='bc', is_edge_norm = True):

    g = load_mat_2_graph(mat_file,atlas)
    print(f'*** \n Graph has {len(g.nodes)} nodes and {len(g.edges)} edges \n***')
    g = find_largest_connected_component(g, show=False)
    print(f'*** \n Connected component has {len(g.nodes)} nodes and {len(g.edges)} edges \n***')

    c = load_core_periphery(g, core_periphery_vec)
    set_edge_type(g)
    selected_nodes = list(g)
    pos = create_nodes_position(atlas)
    pos = {k: v for k, v in pos.items() if k in selected_nodes}
    cmap = cm.get_cmap('bwr', max(c) + 1)
    plt.figure(1, [50, 50])

    if nodes_norm_by == 'deg':
        '''Normalize nodes size by degree'''
        deg = [np.sum(nx.to_numpy_array(g), 0)]
        degnorm = norm_deg(deg,dup=dup)
        nx.draw_networkx_nodes(g, pos, list(g.nodes), node_size=degnorm, cmap=cmap, node_color=c)

    elif nodes_norm_by == 'bc':
        '''Normalize nodes size by betweeness centrality'''
        bc = nx.betweenness_centrality(g,weight='weight')
        nx.draw_networkx_nodes(g, pos, list(g.nodes), node_size=np.asarray(list(bc.values()))*1e6, cmap=cmap, node_color=c)

    else:
        nx.draw_networkx_nodes(g, pos, list(g.nodes), cmap=cmap, node_size=5000, node_color=c)

    #external = [(v, w) for v, w in g.edges if g.edges[v, w]['community'] == 0]
    #internal = [(v, w) for v, w in g.edges if g.edges[v, w]['community'] > 0]
    #internal_color = ['firebrick' for e in internal]
    core_core = [(v, w) for v, w in g.edges if g.edges[v, w]['link'] == 2]
    core_periphery = [(v, w) for v, w in g.edges if g.edges[v, w]['link'] == 1]
    core_core_color = ['firebrick' for e in core_core]
    core_periphery_color = ['slateblue' for e in core_periphery]
    edgewidth = [d['weight'] for (u, v, d) in g.edges(data=True)]

    if is_edge_norm:
        normwidth = norm_edge(edgewidth)

        nx.draw_networkx_edges(g, pos, width=normwidth, alpha=0.2, edgelist=core_periphery, edge_color=core_periphery_color)
        nx.draw_networkx_edges(g, pos, width=normwidth*2, alpha=0.8, edgelist=core_core, edge_color=core_core_color)

    else:
        nx.draw_networkx_edges(g, pos, alpha=0.2, edge_color='silver')
        #nx.draw_networkx_edges(g, pos, alpha=0.9, edgelist=internal, edge_color=internal_color)

    nx.draw_networkx_labels(g, pos, font_size=15, font_weight='bold')
    plt.title(weight_by,fontsize=80)
    plt.savefig(f'G:\data\V7\HCP\core-periphery analysis\Figs\{weight_by}.png')
    plt.show()

def set_edge_type(g):
    '''link is 2 for core-core, 1 for core-periphery & 0 for periphery-periphery'''
    for v, w, in g.edges:
        g.edges[v, w]['link'] = g.nodes[v]['community']+g.nodes[w]['community']


def norm_deg(deg, dup=200):

    degmin = np.min(deg)
    degmax = np.max(deg)
    degnorm = [(20 + dup * (d - degmin) / (degmax - degmin)) ** 2 for d in deg]
    return degnorm


def norm_edge(edgewidth):
    edgewidth = np.asarray(edgewidth)
    p30 = np.nanpercentile(edgewidth,30)
    edgewidth[edgewidth<p30] = np.nan
    wmin = np.nanmin(edgewidth)
    wmax = np.nanmax(edgewidth)
    normwidth = np.asarray([15 * ((w - wmin) / (wmax - wmin)) for w in edgewidth])
    normmean = np.nanmean(normwidth)
    normstd = np.nanstd(normwidth)
    #normwidth[normwidth<[normmean+normstd]] = np.nan
    return normwidth


def run_core_peri_by_group(main_folder, weight_by,th='', nodes_norm='bc'):

    num_file = f'average_num{th}.npy'
    ax_file = f'average_add{th}.npy'
    fa_file = f'average_fa{th}.npy'

    if weight_by == 'fa':
        mat_file = rf'{main_folder}\cm\{fa_file}'
    elif weight_by == 'num':
        mat_file = rf'{main_folder}\cm\{num_file}'
    elif weight_by == 'ax':
        mat_file = rf'{main_folder}\cm\{ax_file}'
    else:
        msg = 'No weight found'
        print(msg)

    for file in os.listdir(rf'{main_folder}\core-periphery analysis'):
        if f'group_level_core_score_759subj{th}' in file:
            core_periphery_file = rf'{main_folder}\core-periphery analysis\{file}'

    mat = sio.loadmat(core_periphery_file)
    if weight_by == 'num':
        core_periphery_vec = mat['c_num'].T
    elif weight_by == 'fa':
        core_periphery_vec = mat['c_fa'].T
    elif weight_by == 'ax':
        core_periphery_vec = mat['c_axsi'].T

    show_topology(mat_file, core_periphery_vec, weight_by, dup=100, nodes_norm_by=nodes_norm, is_edge_norm=True)


def run_core_peri_single_mat(mat_file, core_periphery_file, weight_by, atlas, nodes_norm='bc'):

    core_periphery_vec = sio.loadmat(core_periphery_file)['c'].T
    show_topology(mat_file, core_periphery_vec, weight_by, atlas=atlas, dup=100, nodes_norm_by=nodes_norm, is_edge_norm=True)


if __name__== '__main__':

    main_folder = r'G:\data\V7\HCP'
    #weight_by = 'fa'
    #weight_by = 'num'
    #weight_by = 'ax'
    #run_core_peri_by_group(main_folder, weight_by, '_histmatch_th', nodes_norm='deg')

    atlas = 'yeo7_200'

    #files = glob.glob(f'G:\data\V7\HCP\core-periphery analysis\group_level_core_score_{atlas}*gamma1.25.mat')
    files = ['G:\data\V7\HCP\core-periphery analysis\group_level_core_score_yeo7_200_Num_HistMatch_SC_gamma1.25.mat','G:\data\V7\HCP\core-periphery analysis\group_level_core_score_yeo7_200_ADD_HistMatch_SC_gamma1.25.mat','G:\data\V7\HCP\core-periphery analysis\group_level_core_score_yeo7_200_Dist_HistMatch_SC_gamma1.25.mat','G:\data\V7\HCP\core-periphery analysis\group_level_core_score_yeo7_200_FA_HistMatch_SC_gamma1.25']
    for cp_file in files:
    #     if 'bna' in cp_file:
    #         atlas = 'bna'
    #     elif 'yeo7_200' in cp_file:
    #         atlas = 'yeo7_200'
    #     else:
    #         print('No atlas detected')
    #         continue

        f_parts = cp_file.split(os.sep)[-1].split('.')[0].split('_')[4:-1]
        weight_by = f'{atlas}_{f_parts[-3]}_{f_parts[-2]}_{f_parts[-1]}'
        mat_file = f'{main_folder}{os.sep}cm{os.sep}average_{weight_by}.npy'
        run_core_peri_single_mat(mat_file, cp_file, weight_by, atlas, nodes_norm='deg')

