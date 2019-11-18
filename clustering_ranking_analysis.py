import matplotlib.pyplot as plt
from FT.all_subj import all_subj_names,all_subj_folders
import numpy as np
import pandas as pd
from FT.weighted_tracts import nodes_labels_mega
import networkx as nx
import scipy.io as sio


def all_g_prop():
    subj = all_subj_folders.copy()
    weighted_mat = r'\weighted_mega_wholebrain_plus.npy'
    nonweighted_mat = r'\non-weighted_mega_wholebrain_plus.npy'

    index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlas2nii.txt'
    labels_headers, idx = nodes_labels_mega(index_to_text_file)

    return subj, weighted_mat, nonweighted_mat, labels_headers, idx

def save_df_as_csv(folder_name, rank_table):
    table_file_name = folder_name + r'\clustering_coeff_rank.csv'
    rank_table.to_csv(table_file_name)







'''
    plt.figure(figsize=[15, 15])
    ax0 = plt.subplot(2, 1, 1)
    ax0.set_title('Clustering coefficient distribution (Number of fibers)', fontsize=25)
    ax0.hist(nodes_nw, bins=100)
    ax0.set_ylabel('Frequency', fontsize=15)
    ax0.set_xlabel('Clustering Coefficient', fontsize=15)

    ax1 = plt.subplot(2, 1, 2)
    ax1.set_title('Clustering coefficient distribution (AxCaliber)', fontsize=25)
    ax1.hist(nodes_w, bins=100)
    ax1.set_ylabel('Frequency', fontsize=15)
    ax1.set_xlabel('Clustering Coefficient', fontsize=15)

    plt.show()
'''

if __name__ == '__main__':
    subj, weighted_mat, nonweighted_mat, labels_headers, idx =all_g_prop()
    #nodes_nw= []
    #nodes_w = []

    nodes_nw = np.zeros([len(subj),len(idx)])
    nodes_w = np.zeros([len(subj),len(idx)])

    for i,s in enumerate(subj):
        folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep' + s

        # non-weighted:
        mat_file_name = folder_name + nonweighted_mat
        mat = np.load(mat_file_name)
        mat[mat < 0] = 0
        mat[mat > 1] = 0

        G = nx.from_numpy_array(mat)
        clustering_nw_vals = nx.clustering(G, weight='weight')
        nw = pd.DataFrame.from_dict(clustering_nw_vals, orient='index')

        # weighted:
        mat_file_name = folder_name + weighted_mat
        mat = np.load(mat_file_name)
        mat[mat < 0] = 0
        mat[mat > 1] = 0

        G = nx.from_numpy_array(mat)
        clustering_w_vals = nx.clustering(G, weight='weight')
        w = pd.DataFrame.from_dict(clustering_w_vals, orient='index')

        rank_table = pd.concat([nw, w], axis=1)
        rank_table.columns = ['non-weighted_vals', 'weighted_vals']
        rank_table['non-weighted_ranks'] = rank_table['non-weighted_vals'].rank().astype('int64')
        rank_table['weighted_ranks'] = rank_table['weighted_vals'].rank().astype('int64')

        rank_table['cortex_part'] = labels_headers
        rank_table['mutual'] = (rank_table['weighted_ranks'] + rank_table['non-weighted_ranks'])
        rank_table['mutual_rank'] = rank_table['mutual'].rank().astype('int64')

        save_df_as_csv(folder_name, rank_table)

        nodes_nw[i,:] = np.asarray(rank_table['non-weighted_ranks'])
        nodes_w[i,:] = np.asarray(rank_table['weighted_ranks'])

        #nodes_nw = nodes_nw + list(rank_table['non-weighted_vals'])
        #nodes_w = nodes_w + list(rank_table['weighted_vals'])

    nw_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\Testings\clus_nw.mat'
    w_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\Testings\clus_w.mat'
    sio.savemat(nw_name, {'nw_clustering_coeff_mat': nodes_nw})
    sio.savemat(w_name, {'w_clustering_coeff_mat': nodes_w})

    np.save(r'C:\Users\Admin\my_scripts\Ax3D_Pack\Testings\clus_nw',nodes_nw)
    np.save(r'C:\Users\Admin\my_scripts\Ax3D_Pack\Testings\clus_w',nodes_w)


    plt.figure(figsize=[30, 15])
    ax0 = plt.subplot(2, 1, 1)
    ax0.bar(idx,np.median(nodes_nw,axis=0))
    ax0.set_title('Clustering coefficient median ranks (Number of fibers) \n (inter-subject)', fontsize=25)
    ax0.set_ylabel('Median clustering Coefficient rank',fontsize=15)
    ax0.set_xlabel('Cortex nodes',fontsize=15)
    ax0.set_xticks(idx)

    ax1 = plt.subplot(2, 1, 2)
    ax1.bar(idx,np.median(nodes_w,axis=0))
    ax1.set_title('Clustering coefficient median ranks (AxCaliber) \n (inter-subject)', fontsize=25)
    ax1.set_ylabel('Median clustering Coefficient rank', fontsize=15)
    ax1.set_xlabel('Cortex nodes',fontsize=15)
    ax1.set_xticks(idx)
    ax1.set_xticklabels(labels_headers)
    ax1.tick_params(axis='x', pad=8.0, labelrotation=90, labelsize=11)
    plt.show()

    plt.figure(figsize=[30, 15])
    ax0 = plt.subplot(1,2, 1)
    ax0.scatter(np.median(nodes_nw[0:50],axis=0),np.median(nodes_nw[50::],axis=0))
    ax0.set_title('Clustering coefficient median ranks (Number of fibers) \n Left Vs. Right hemispheres', fontsize=25)
    ax0.set_ylabel('Right hemisphere',fontsize=15)
    ax0.set_xlabel('Left hemisphere',fontsize=15)

    ax1 = plt.subplot(1,2, 2)
    ax1.scatter(np.median(nodes_w[0:50],axis=0),np.median(nodes_w[50::],axis=0))
    ax1.set_title('Clustering coefficient median ranks (AxCaliber) \n Left Vs. Right hemispheres', fontsize=25)
    ax1.set_ylabel('Right hemisphere',fontsize=15)
    ax1.set_xlabel('Left hemisphere',fontsize=15)
    plt.show()
'''
    plt.figure(figsize=[15,15])
    ax0 = plt.subplot(2,1,1)
    ax0.set_title('Clustering coefficient distribution (Number of fibers)',fontsize=25)
    ax0.hist(nodes_nw, bins = 100)
    ax0.set_ylabel('Frequency',fontsize=15)
    ax0.set_xlabel('Clustering coefficient level',fontsize=15)


    ax1 = plt.subplot(2,1,2)
    ax1.set_title('Clustering coefficient distribution (AxCaliber)', fontsize=25)
    ax1.hist(nodes_w, bins = 100)
    ax1.set_ylabel('Frequency',fontsize=15)
    ax1.set_xlabel('Clustering coefficient level',fontsize=15)

    plt.show()
        fig, ax = plt.subplots(figsize=[30, 15])
        x = np.asarray(range(len(rank_table.index)))
        s1 = ax.bar(x - 0.3 / 2, rank_table['non-weighted_ranks'], 0.3, label='Non-weighted')
        s2 = ax.bar(x + 0.3 / 2, rank_table['weighted_ranks'], 0.3, label='Weighted')
        ax.set_ylabel('Clustering Coefficient rank')
        ax.set_xticks(x)
        ax.set_xticklabels(rank_table['cortex_part'])
        ax.tick_params(axis='x', pad=8.0, labelrotation=90, labelsize=11)

        ax.legend()
        for rect in s1:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        for rect in s2:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        fig.tight_layout()
        plt.show()

        plt.bar(rank_table.index, 'mutual_rank','non-weighted_ranks', 'weighted_ranks', s='mutual_rank', data=rank_table)
'''