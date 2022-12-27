from HCP_network_analysis.core_periphery_representation import load_mat_2_graph
from network_analysis.topology_rep import *

atlas='bna'
th='_histmatch_th'
folder_name = r'G:\data\V7\HCP\cm'
num_file = f'{atlas}_average_num{th}.npy'
ax_file = f'{atlas}_average_add{th}.npy'
fa_file = f'{atlas}_average_fa{th}.npy'

fa_mat_file = rf'{folder_name}\{fa_file}'
num_mat_file = rf'{folder_name}\{num_file}'
add_mat_file = rf'{folder_name}\{ax_file}'

g_fa = load_mat_2_graph(fa_mat_file, atlas, False)
print(f'*** \n Graph has {len(g_fa.nodes)} nodes and {len(g_fa.edges)} edges \n***')
g_fa = find_largest_connected_component(g_fa, show=False)
print(f'*** \n Connected component has {len(g_fa.nodes)} nodes and {len(g_fa.edges)} edges \n***')
#sig_fa = nx.sigma(g_fa)
omg_fa = nx.omega(g_fa)

g_num = load_mat_2_graph(num_mat_file, atlas, False)
print(f'*** \n Graph has {len(g_num.nodes)} nodes and {len(g_num.edges)} edges \n***')
g_num = find_largest_connected_component(g_num, show=False)
print(f'*** \n Connected component has {len(g_num.nodes)} nodes and {len(g_num.edges)} edges \n***')
#sig_num = nx.sigma(g_num)
omg_num = nx.omega(g_num)

g_add = load_mat_2_graph(add_mat_file, atlas, False)
print(f'*** \n Graph has {len(g_add.nodes)} nodes and {len(g_add.edges)} edges \n***')
g_add = find_largest_connected_component(g_add, show=False)
print(f'*** \n Connected component has {len(g_add.nodes)} nodes and {len(g_add.edges)} edges \n***')
#sig_add = nx.sigma(g_add)
omg_add = nx.omega(g_add)

import matplotlib.pyplot as plt
#labels = ['sigma #Streamlines', 'omega #Stramlines', 'sigma ADD', 'omega ADD', 'sigma FA', 'omega FA']
labels = ['#Stramlines', 'ADD','FA']
#colors = ['r','b','r','b','r','b']
plt.bar([0,1,2],[omg_num,omg_add,omg_fa],labels=labels)

