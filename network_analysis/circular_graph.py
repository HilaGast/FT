import networkx as nx
import nxviz as nv
import matplotlib.pyplot as plt
import numpy as np
from all_subj import index_to_text_file
from weighted_tracts import nodes_labels_yeo7,nodes_labels_aal3
from network_analysis.specific_functional_yeo7networks import network_id_list

atlas = 'yeo7'
network = 'sommot'
side = 'both'
if atlas == 'yeo7':
    labels_headers, idx = nodes_labels_yeo7(index_to_text_file)
    n = len(idx)
    start = int(n/2-1)
    cont = start+1
    id = idx[start::-1]+idx[cont::]
elif atlas == 'aal3':
    labels_headers, idx = nodes_labels_aal3(index_to_text_file)
    n = len(idx)
    id = list(range(n))
    start = 77 #end of left sided areas
    cont = start+1
    id = id[start::-1]+id[cont::]

nodes_labels={}
for l,i in zip(labels_headers,id):
    lparts = l.split('_')
    #nodes_labels[i] = '_'.join(lparts[1:])
    nodes_labels[i]=f'{lparts[1]}_{lparts[2]}_{lparts[-1]}'
    #nodes_labels[i]=l
pmat_name = r'C:\Users\Admin\Desktop\balance\eo_pval_aal_norm_add.npy'
tmat_name = r'C:\Users\Admin\Desktop\balance\eo_ttest_aal_norm_add.npy'
pmat = np.load(pmat_name)
mat = np.load(tmat_name)
pmat = pmat[id]
pmat= pmat[:,id]
mat=mat[id]
mat=mat[:,id]
#nodes_labels = [nodes_labels[node] for node in id]

ii = network_id_list(network,side)
#pmat = pmat[ii]
#pmat = pmat[:,ii]

#mat=mat[ii]
#mat=mat[:,ii]

#nodes_labels = {node:nodes_labels[node] for node in nodes_labels.keys() &  ii}

#mat[(np.abs(mat)<3)]=0
pmat[np.isnan(pmat)] = 0
mat[pmat>0.05] = 0
mat[np.isnan(mat)]=0
G = nx.from_numpy_array(mat)
for v, w in G.edges:
    if G.edges[v, w]['weight'] > 0:
        G.edges[v, w]['sign'] = 'pos'
    elif G.edges[v, w]['weight'] < 0:
        G.edges[v, w]['sign'] = 'neg'

for n in G.nodes():
    G.nodes[n]['size']=3
    if n>len(G)/2:
        G.nodes[n]['side'] = "left"
    else:
        G.nodes[n]['side'] = "right"

G = nx.relabel_nodes(G,nodes_labels)
G1=G.subgraph(ii)

nodes_list = [nodes_labels[id[idx[node]]] for node in ii]

c = nv.circos(G)
c = nv.circos(G.subgraph(nodes_list), node_labels=True, edge_width='weight',
               edge_color='weight',edgeprops = {"facecolor": "none", "alpha": 1},
               fig_size = (15,15),node_label_layout='rotation',group_label_offset=4,
               fontsize=7, node_size='size')
c.draw()

plt.show()

