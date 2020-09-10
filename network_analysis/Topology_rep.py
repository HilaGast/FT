import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from weighted_tracts import *
from network_analysis.create_labels_centroid_2d import create_nodes_position

def find_largest_connected_component(g, show=False):
    pos = nx.layout.spring_layout(g)
    Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    G0 = g.subgraph(Gcc[0])

    if show:
        plt.figure(figsize=(20, 20))
        nx.draw(G0, pos, node_size=[g.degree],
            with_labels=True,
            node_color='r',
            edge_color='k',
            width=0.1, font_size=5)

        plt.show()
    return G0


def find_k_cliques_modules(id, idx, mat_weights, th):
    from networkx.algorithms.community import k_clique_communities
    matbin = mat_weights > th
    matbin = matbin * 1
    matth = matbin[id]
    matth = matth[:, id]
    a = np.sort(idx)
    gnew = nx.Graph()
    gnew.add_nodes_from(a)  # gnew is non-weighted graph, edges masked using th
    g = nx.from_numpy_array(matth)
    gnew.add_edges_from(g.edges)
    G0 = find_largest_connected_component()
    c = list(k_clique_communities(G0, 4))

    return c


def set_node_community(g, comunity_dict):
    '''Add community to node attributes'''
    for n,c in comunity_dict.items():
            # Add 1 to save 0 for external edges
            g.nodes[n]['community'] = c + 1


def set_edge_community(g):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in g.edges:
        if g.nodes[v]['community'] == g.nodes[w]['community']:
            # Internal edge, mark with community
            g.edges[v, w]['community'] = g.nodes[v]['community']
        else:
            # External edge, mark as 0
            g.edges[v, w]['community'] = 0


def find_louvain_modules(g,r,deg, show=True):
    import community
    louvain_partition = community.best_partition(g, weight='weight',resolution=r)
    print(f'*** \n Found {len(set(louvain_partition.values()))} modules\n***')

    if show:
        import matplotlib.cm as cm
        set_node_community(g, louvain_partition)
        set_edge_community(g)
        #pos = nx.spring_layout(g)
        pos = create_nodes_position()
        selected_nodes = list(g)
        pos = {k:v for k,v in pos.items() if k in selected_nodes}
        cmap = cm.get_cmap('gist_rainbow', max(louvain_partition.values()) + 1)
        plt.figure(1, [50, 50])
        nx.draw_networkx_nodes(g, pos, louvain_partition.keys(), node_size=deg,
                               cmap=cmap, node_color=list(louvain_partition.values()))
        edgewidth = [d['weight'] for (u, v, d) in g.edges(data=True)]
        wmin = np.min(edgewidth)
        wmax = np.max(edgewidth)
        normwidth = np.asarray([1+30*((w-wmin)/(wmax-wmin))**8 for w in edgewidth])

        #normwidth[edgewidth<np.percentile(edgewidth,80)] = 0.00

        external = [(v, w) for v, w in g.edges if g.edges[v, w]['community'] == 0]
        internal = [(v, w) for v, w in g.edges if g.edges[v, w]['community'] > 0]
        internal_color = ['black' for e in internal]

        nx.draw_networkx_edges(g, pos, width=normwidth, alpha=0.2, edge_color='silver')
        nx.draw_networkx_edges(g, pos, width=normwidth, alpha=1, edgelist=internal, edge_color=internal_color)

        nx.draw_networkx_labels(g,pos,font_size=50)
        plt.show()
        plt.hist(normwidth,bins=20)
        plt.show()
    # to assess modularity:
    # modularity2 = community.modularity(louvain_partition, G0, weight='weight')
    # print("The modularity Q based on networkx is {}".format(modularity2))


if __name__ == '__main__':
    modularity_methods=['kcliques','louvain']
    method=modularity_methods[1]
    #mat_file = rf'{subj_folder}{all_subj_folders[7]}\weighted_mega_wholebrain_4d_labmask_nonnorm.npy'
    #mat_file = 'F:\Hila\Ax3D_Pack\mean_vals\FA\mean_weighted_mega_wholebrain_4d_labmask_nonnorm.npy'
    mat_file = r'F:\Hila\Ax3D_Pack\mean_vals\aal3_atlas\mean_non-weighted_mega_wholebrain_4d_labmask_aal3_nonnorm.npy'
    mat = np.load(mat_file)
    #if "FA" in mat_file:
    #    mat = mat/100
    idx = nodes_labels_aal3(index_to_text_file)[1]
    id = np.argsort(idx)
    mat_weights = mat[id]
    mat_weights = mat_weights[:, id]
    mat = np.zeros((mat_weights.shape[0]+1,mat_weights.shape[1]+1))
    mat[1:,1:]=mat_weights
    g = nx.from_numpy_array(mat)
    print(f'*** \n Graph has {len(g.nodes)} nodes and {len(g.edges)} edges \n***')

    if method=='louvain':
        r=1
        g_connected = find_largest_connected_component(g,show=False)
        print(f'*** \n Connected component has {len(g_connected.nodes)} nodes and {len(g_connected.edges)} edges \n***')
        deg = [np.sum(nx.to_numpy_array(g_connected),0)]
        degmin = np.min(deg)
        degmax = np.max(deg)
        degnorm = [(20+200*(d-degmin)/(degmax-degmin))**2 for d in deg]
        #plt.hist(degnorm)
        #plt.show()
        find_louvain_modules(g_connected,r,degnorm)








