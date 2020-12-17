import networkx as nx
import numpy as np
from weighted_tracts import nodes_labels_yeo7
from all_subj import index_to_text_file
import matplotlib.pyplot as plt
import scipy.stats as stats
from network_analysis.community_based_topology import load_mat_2_graph,find_largest_connected_component

mat_file = r'F:\Hila\Ax3D_Pack\mean_vals\yeo7_200\mean_non-weighted_wholebrain_4d_labmask_yeo7_200_nonnorm.npy'
#mat_file = r'F:\Hila\Ax3D_Pack\V6\after_file_prep\YA_lab_Yaniv_Subj000603_20190822_1529\weighted_wholebrain_4d_labmask_yeo7_200_nonnorm.npy'
mat = np.load(mat_file)

idx = nodes_labels_yeo7(index_to_text_file)[1]
id = np.argsort(idx)
mat_weights = mat[id]
mat_weights = mat_weights[:, id]
mat = np.zeros((mat_weights.shape[0] + 1, mat_weights.shape[1] + 1))
mat[1:, 1:] = mat_weights
g = nx.from_numpy_array(mat)

#g, labels = load_mat_2_graph(mat_file)
print(f'*** \n Graph has {len(g.nodes)} nodes and {len(g.edges)} edges \n***')
g = find_largest_connected_component(g, show=False)
print(f'*** \n Connected component has {len(g.nodes)} nodes and {len(g.edges)} edges \n***')
#label_dict = set_node_label(g, labels)


ebc = nx.edge_betweenness_centrality(g,weight=None,normalized=False)
# edgeweight = {(u,v):d['weight'] for (u, v, d) in g.edges(data=True) if (u,v) in ebc.keys()}

edgeweight = [d['weight'] for (u,v,d) in g.edges(data=True) if (u,v) in ebc.keys()]
ebc_list = [ebc[(u,v)] for (u,v,d) in g.edges(data=True) if (u,v) in ebc.keys()]
ebc_list = np.asarray(ebc_list)
edgeweight_list = np.asarray(edgeweight)
#highebc = ebc_list > 2*np.median(ebc_list)

#plt.scatter(-1*np.log(ebc_list[highebc]),edgeweight_list[highebc])

ebc_rank = stats.rankdata(ebc_list)
edgeweight_rank = stats.rankdata(edgeweight_list)
plt.scatter(edgeweight_list,ebc_rank,c='black',marker= '.',s=3)
plt.show()

