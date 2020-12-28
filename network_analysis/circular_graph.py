import networkx as nx
from nxviz.plots import CircosPlot
import matplotlib.pyplot as plt
import numpy as np
from all_subj import index_to_text_file
from weighted_tracts import nodes_labels_yeo7,nodes_labels_aal3

labels_headers, idx = nodes_labels_yeo7(index_to_text_file)
n = len(idx)
start = int(n/2-1)
cont = start+1
id = idx[start::-1]+idx[cont::]
nodes_labels={}
for l,i in zip(labels_headers,id):
    lparts = l.split('_')
    nodes_labels[i]=f'{lparts[1]}_{lparts[2]}_{lparts[-1]}'
    #nodes_labels[i]=f'{lparts[-1]}_{lparts[0]}'
pmat_name = r'C:\Users\Admin\Desktop\balance_plasticity\yeo\eo_pval_axcaliber.npy'
tmat_name = r'C:\Users\Admin\Desktop\balance_plasticity\yeo\eo_ttest_axcaliber.npy'
pmat = np.load(pmat_name)
mat = np.load(tmat_name)
pmat = pmat[id]
pmat= pmat[:,id]
mat=mat[id]
mat=mat[:,id]
#mat[(np.abs(mat)<3)]=0
pmat[np.isnan(pmat)] = 0
mat[pmat>0.005] = 0
mat[np.isnan(mat)]=0
G = nx.from_numpy_array(mat)
for v, w in G.edges:
    if G.edges[v, w]['weight'] > 0:
        G.edges[v, w]['sign'] = 'pos'
    elif G.edges[v, w]['weight'] < 0:
        G.edges[v, w]['sign'] = 'neg'

for n in G.nodes():
    G.node[n]['size']=3
    if n>len(G)/2:
        G.node[n]['side'] = "left"
    else:
        G.node[n]['side'] = "right"

G = nx.relabel_nodes(G,nodes_labels)

c = CircosPlot(G,node_labels=True, edge_width='weight',
               edge_color='weight',edgeprops = {"facecolor": "none", "alpha": 1},
               fig_size = (15,15),node_label_layout='rotation',group_label_offset=4,
               fontsize=7, node_size='size')
c.draw()


#posa = [(v,w) for v,w in G.edges if G.edges[v,w]['weight']>0]
#nega = [(v,w) for v,w in G.edges if G.edges[v,w]['weight']<0]


#nx.drawing.draw_circular(G, with_labels = True, edge_list=posa, edge_color='red', node_color= 'gray', node_size=100)
#nx.drawing.draw_circular(G, with_labels = True, edge_list=nega, edge_color='blue', node_color= 'gray', node_size=100)

plt.show()

