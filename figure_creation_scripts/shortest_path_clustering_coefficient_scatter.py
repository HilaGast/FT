import os, glob
import numpy as np
from network_analysis.global_network_properties import get_mean_shortest_path, get_clustering_coefficient
from network_analysis.nodes_network_properties import get_local_efficiency
import matplotlib.pyplot as plt


subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
add_sp=[]
num_sp=[]
fa_sp=[]

add_cc=[]
num_cc=[]
fa_cc=[]

th = 'HistMatch'
atlas = 'yeo7_200'
for sl in subj_list[:100]:

    add_cm = np.load(f'{sl}cm{os.sep}{atlas}_ADD_{th}_SC_cm_ord.npy')
    add_sp.append(np.nanmean(get_mean_shortest_path(add_cm)))
    add_cc.append(get_clustering_coefficient(add_cm))


    num_cm = np.load(f'{sl}cm{os.sep}{atlas}_Num_{th}_SC_cm_ord.npy')
    num_sp.append(np.nanmean(get_mean_shortest_path(num_cm)))
    num_cc.append(get_clustering_coefficient(num_cm))

    fa_cm = np.load(f'{sl}cm{os.sep}{atlas}_FA_{th}_SC_cm_ord.npy')
    fa_sp.append(np.nanmean(get_mean_shortest_path(fa_cm)))
    fa_cc.append(get_clustering_coefficient(fa_cm))

plt.scatter(num_sp, num_cc, color=[0.2, 0.7, 0.6], s=10)
plt.scatter(add_sp, add_cc, color=[0.8, 0.5, 0.3], s=10)
plt.scatter(fa_sp, fa_cc, color=[0.3, 0.3, 0.5], s=10)

plt.legend(['Num', 'ADD', 'FA'])

plt.xlabel('Average Shortest Path')
plt.ylabel('Average Clustering Coefficient')
plt.show()
