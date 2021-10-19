import networkx as nx
import nxviz as nv
import matplotlib.pyplot as plt
import numpy as np
from all_subj import index_to_text_file
from weighted_tracts import nodes_labels_yeo7,nodes_labels_aal3
from network_analysis.specific_functional_yeo7networks import network_id_list


def calc_t_mat(tmat_name, pmat_name, pmat_cutoff):
    pmat = np.load(pmat_name)
    mat = np.load(tmat_name)

    pmat[np.isnan(pmat)] = 0

    mat[pmat > pmat_cutoff] = 0

    return mat


def mat_2_graph(mat):
    mat[np.isnan(mat)] = 0
    G = nx.from_numpy_array(mat)

    return G


def create_labels_dict(atlas = 'yeo7'):

    if atlas == 'yeo7':
        labels_headers, idx = nodes_labels_yeo7(index_to_text_file)
        labels_dict={}
        sides_dict={}
        net_dict = {}
        for l,i in zip (labels_headers,idx):
            lparts = l.split('_')
            labels_dict[i]=f'{lparts[1]}_{lparts[2]}_{lparts[-1]}'
            net_dict[i] = f'{lparts[1]}_{lparts[2]}'
            #net_dict[i] = f'{lparts[2]}'

            if '_LH_' in l:
                sides_dict[i]='LH'
            elif '_RH_' in l:
                sides_dict[i]='RH'
            else:
                sides_dict[i]='C'
        return sides_dict, labels_dict, net_dict

    elif atlas == 'aal3':
        labels_headers, idx = nodes_labels_aal3(index_to_text_file)
        labels_dict={}
        sides_dict={}
        for l,i in zip (labels_headers,idx):
            labels_dict[i]=l

            if '_L' in l:
                sides_dict[i]='LH'
            elif '_R' in l:
                sides_dict[i]='RH'
            else:
                sides_dict[i]='C'

        return sides_dict, labels_dict


def choose_subnetwork(G,side='both',network='sommot'):

    ii = network_id_list(network, side)
    G1 = G.subgraph(ii)

    return G1


def label_nodes_by_dict(G,Gdict,label_title):
    for n in G.nodes():
        G.nodes[n][label_title] = Gdict[n]

    return G


def label_edges_sign(G):
    for v, w in G.edges:
        if G.edges[v, w]['weight'] > 0:
            G.edges[v, w]['sign'] = 'pos'
        elif G.edges[v, w]['weight'] < 0:
            G.edges[v, w]['sign'] = 'neg'

    return G


def define_nodes_properties(G, nodes_size):
    for n in G.nodes():
        G.nodes[n]['size'] = nodes_size

    return G


def draw_circular_graph(G,show=True,save=True):
    from nxviz.plots import aspect_equal, despine
    from nxviz import nodes, edges, annotate
    plt.figure(figsize=(10,10))
    ax=plt.gca()
    G_nt = nv.utils.node_table(G)
    G_et = nv.utils.edge_table(G)
    G_et = G_et[G_et['source'] <= G_et['target']]
    pos = nv.layouts.circos(G_nt, sort_by='side')
    node_colors = nv.nodes.node_colors(G_nt,'side')
    node_alpha = {key: 1 for key in G.nodes}
    node_size = {key: 1 for key in G.nodes}
    node_patches = nv.nodes.node_glyphs(
        G_nt, pos, node_color=node_colors,
        alpha=node_alpha, size=node_size)
    for patch in node_patches:
        ax.add_patch(patch)

    edge_color = {}
    for key, props in G_et.iterrows():
        source = G.nodes[props.source]
        target = G.nodes[props.target]
        if source['labels'] == target['labels']:
            source_node_table_index = G_nt.index.get_loc(props.source)
            edge_color[key] = node_colors[source_node_table_index]
        else:
            edge = G.edges[(props.source, props.target)]
            edge_color[key] = 'red' if edge['sign'] == 'pos' else 'blue'

    edge_weight = G_et['weight']
    edge_weight = abs(edge_weight)
    edge_weight = edge_weight - min(edge_weight)
    edge_weight = edge_weight / max(edge_weight)
    edge_weight = np.power(edge_weight, 2) * 5
    edge_alpha = 0.5 + 0.5 * edge_weight / max(edge_weight)

    edge_patches = nv.lines.circos(
        G_et, pos, edge_color=edge_color,
        alpha=edge_alpha, lw=edge_weight, aes_kw={"fc": "none"})

    for patch in edge_patches:
        ax.add_patch(patch)


    nv.plots.rescale(G)


    #ax = nv.circos(G, group_by = 'net', node_color_by = 'side',edge_alpha_by = 'weight')
    #pos = nodes.circos(G, group_by='net',color_by='side')#, size_scale='size',edge_alpha_by="weight",edge_color_by='sign')
    #edges.draw(G,pos,lines_func='circos',color_by=G['sign'],alpha_by=G['weight'])

    annotate.circos_group(G,group_by="net",radius_offset=1)
    #annotate.edge_colormapping(G,color_by="sign")


    despine()
    aspect_equal()

    if show:
        plt.show()




