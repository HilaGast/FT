import matplotlib.pyplot as plt
from FT.all_subj import all_subj_names,all_subj_folders
import numpy as np
import pandas as pd
from FT.weighted_tracts import nodes_labels_mega
def all_g_prop():
    subj = all_subj_folders.copy()
    names = all_subj_names
    #weighted_mat = r'\weighted_mega_wholebrain_cortex_nonnorm.npy'
    #nonweighted_mat = r'\non-weighted_mega_wholebrain_cortex_nonnorm.npy'
    weighted_mat = r'\weighted_mega_wholebrain_cortex_nonnorm.npy'
    nonweighted_mat = r'\non-weighted_mega_wholebrain_cortex_nonnorm.npy'
    nodes_nw=[]
    nodes_w = []
    for s,n in zip(subj,names):

        folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep' + s

        #non-weighted:
        mat_file_name = folder_name+nonweighted_mat
        mat = np.load(mat_file_name)
        #mat[mat > 1] = 0
        mat[mat < 0] = 0
        deg_nw_vals = np.sum(mat, axis=0)
        nw = pd.Series(deg_nw_vals)

        #weighted:
        mat_file_name = folder_name+weighted_mat
        mat = np.load(mat_file_name)
        #mat[mat > 1] = 0
        mat[mat < 0] = 0
        deg_w_vals = np.sum(mat, axis=0)
        w = pd.Series(deg_w_vals)

        rank_table = pd.concat([nw,w],axis=1)
        rank_table.columns = ['non-weighted_vals','weighted_vals']
        rank_table['non-weighted_ranks'] = rank_table['non-weighted_vals'].rank().astype('int64')
        rank_table['weighted_ranks'] = rank_table['weighted_vals'].rank().astype('int64')


        index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlascortex2nii.txt'
        labels_headers, idx = nodes_labels_mega(index_to_text_file)

        rank_table['cortex_part'] =  labels_headers
        rank_table['mutual'] = (rank_table['weighted_ranks']+rank_table['non-weighted_ranks'])
        rank_table['mutual_rank'] = rank_table['mutual'].rank().astype('int64')

        #rank_table.sort_values("mutual_rank", inplace = True)
        table_file_name = folder_name + r'\node_degree_rank.csv'
        #rank_table.to_csv(table_file_name)
        nodes_nw = nodes_nw + list(rank_table['non-weighted_vals'])
        nodes_w = nodes_w + list(rank_table['weighted_vals'])


    plt.figure(figsize=[15,15])
    ax0 = plt.subplot(2,1,1)
    ax0.set_title('Node degree distribution (Number of fibers)',fontsize=25)
    ax0.hist(nodes_nw, bins = 100)
    ax0.set_ylabel('Frequency',fontsize=15)
    ax0.set_xlabel('Node degree level',fontsize=15)


    ax1 = plt.subplot(2,1,2)
    ax1.set_title('Node degree distribution (AxCaliber)', fontsize=25)
    ax1.hist(nodes_w, bins = 100)
    ax1.set_ylabel('Frequency',fontsize=15)
    ax1.set_xlabel('Node degree level',fontsize=15)

    plt.show()




if __name__ == '__main__':
    all_g_prop()





