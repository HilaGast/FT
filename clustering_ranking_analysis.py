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
    id = np.argsort(idx)

    return subj, weighted_mat, nonweighted_mat, labels_headers, id

def save_df_as_csv(folder_name, rank_table):
    table_file_name = folder_name + r'\clustering_coeff_rank.csv'
    rank_table.to_csv(table_file_name)





if __name__ == '__main__':
    subj, weighted_mat, nonweighted_mat, labels_headers, id =all_g_prop()
    #nodes_nw= []
    #nodes_w = []

    nodes_nw = np.zeros([len(subj),len(id)])
    nodes_w = np.zeros([len(subj),len(id)])

    for i,s in enumerate(subj):
        folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep' + s

        # non-weighted:
        mat_file_name = folder_name + nonweighted_mat
        mat = np.load(mat_file_name)
        mat = mat[id]
        mat = mat[:,id]
        mat[mat < 0] = 0
        mat[mat > 1] = 0

        G = nx.from_numpy_array(mat)
        clustering_nw_vals = nx.clustering(G, weight='weight')
        nw = pd.DataFrame.from_dict(clustering_nw_vals, orient='index')

        # weighted:
        mat_file_name = folder_name + weighted_mat
        mat = np.load(mat_file_name)
        mat = mat[id]
        mat = mat[:,id]
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

        #save_df_as_csv(folder_name, rank_table)

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

