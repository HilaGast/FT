import matplotlib.pyplot as plt
from FT.all_subj import all_subj_names
import numpy as np

def all_g_prop():
    import networkx as nx
    subj = all_subj_names.copy()
    weighted_mat = r'\weighted_mega_wholebrain_cortex_nonnorm.npy'
    nonweighted_mat = r'\non-weighted_mega_wholebrain_cortex_nonnorm.npy'
    data = np.empty([2,len(subj),3])
    i=0
    for s in subj:

        folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5' + s
        #non-weighted:
        mat_file_name = folder_name+nonweighted_mat
        mat = np.load(mat_file_name)
        #mat[mat > 1] = np.inf
        mat[mat < 0] = 0
        #opmat = 1-mat
        #opmat[opmat<0] = 0
        #mat[mat > 1] = 100
        deg = np.sum(mat[np.isfinite(mat)], axis=0)
        mean_deg = np.nanmean(deg)
        G = nx.from_numpy_array(mat)
        mean_shortest_path = nx.average_shortest_path_length(G,weight='weight')
        #G = nx.from_numpy_array(opmat)
        mean_clustering = nx.average_clustering(G, weight='weight')
        data[0,i,0] = mean_deg
        data[0,i,1] = mean_shortest_path
        data[0,i,2] = mean_clustering

        #weighted:
        mat_file_name = folder_name+weighted_mat
        mat = np.load(mat_file_name)
        #mat[mat > 1] = np.inf
        mat[mat < 0] = 0
        #opmat = 1-mat
        #opmat[opmat<0] = 0
        #mat[mat > 1] = 100
        deg = np.sum(mat[np.isfinite(mat)], axis=0)
        mean_deg = np.nanmean(deg)
        G = nx.from_numpy_array(mat)
        mean_shortest_path = nx.average_shortest_path_length(G,weight='weight')
        #G = nx.from_numpy_array(opmat)
        mean_clustering = nx.average_clustering(G, weight='weight')
        data[1,i,0] = mean_deg
        data[1,i,1] = mean_shortest_path
        data[1,i,2] = mean_clustering
        print(i)
        i+=1
    return data

if __name__ == '__main__':
    data = all_g_prop()

    plt.figure(figsize=[12,6])

    ax0 = plt.subplot(1,3,1)
    ax0.set_title('Average Nodes Degree')
    #ax0.scatter(data[0,:,0]/data[0,:,0].max(),data[1,:,0]/data[1,:,0].max())
    ax0.scatter(data[0,:,0],data[1,:,0])



    ax1 = plt.subplot(1,3,2)
    ax1.set_title('Average Shortest Path')
    #ax1.scatter(data[0,:,1]/data[0,:,1].max(),data[1,:,1]/data[1,:,1].max())
    ax1.scatter(data[0,:,1],data[1,:,1])


    ax2=plt.subplot(1,3,3)
    ax2.set_title('Average Clustering Coefficient')
    #ax2.scatter(data[0,:,2]/data[0,:,2].max(),data[1,:,2]/data[1,:,2].max())
    ax2.scatter(data[0, :, 2], data[1, :, 2])


ax0.set_ylabel('AxCaliber weighted graph', fontsize=12)
ax1.set_xlabel('Number of tracts weighted graph', fontsize=12)
plt.show()
