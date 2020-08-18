import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from weighted_tracts import *

mat_file = r'C:\Users\hila\data\subj\YA_lab_Yaniv_001555_20200510_1058\weighted_mega_wholebrain_4d_labmask_FA_nonnorm.npy'
mat = np.load(mat_file)
mat = mat/100
index_to_text_file = r'C:\Users\hila\data\megaatlas\megaatlas2nii.txt'
idx = nodes_labels_mega(index_to_text_file)[1]
id = np.argsort(idx)
mat_weights = mat[id]
mat_weights = mat_weights[:, id]

g = nx.from_numpy_array(mat)

pos = nx.layout.spring_layout(g)

Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
G0 = g.subgraph(Gcc[0])
plt.figure(figsize=(20, 20))
nx.draw(G0, node_size=[g.degree],
                       with_labels=True,
                        node_color='r',
                       edge_color='k',
                       width=0.1,font_size=5)

plt.show()
