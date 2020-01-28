import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from FT.clustering_ranking_analysis import all_g_prop







if __name__ == '__main__':
    subj, weighted_mat, nonweighted_mat, labels_headers, id =all_g_prop()
    nodes_nw = np.zeros([len(subj),len(id)])
    nodes_w = np.zeros([len(subj),len(id)])
    for i,s in enumerate(subj):

        folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep' + s

        #non-weighted:
        mat_file_name = folder_name+nonweighted_mat
        mat = np.load(mat_file_name)
        mat = mat[id]
        mat = mat[:,id]
        mat[mat > 1] = 0
        mat[mat < 0] = 0
        deg_nw_vals = np.sum(mat, axis=0)
        nw = pd.Series(deg_nw_vals)

        #weighted:
        mat_file_name = folder_name+weighted_mat
        mat = np.load(mat_file_name)
        mat = mat[id]
        mat = mat[:,id]
        mat[mat > 1] = 0
        mat[mat < 0] = 0
        deg_w_vals = np.sum(mat, axis=0)
        w = pd.Series(deg_w_vals)

        rank_table = pd.concat([nw,w],axis=1)
        rank_table.columns = ['non-weighted_vals','weighted_vals']
        rank_table['non-weighted_ranks'] = rank_table['non-weighted_vals'].rank().astype('int64')
        rank_table['weighted_ranks'] = rank_table['weighted_vals'].rank().astype('int64')


        index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlas2nii.txt'
        #labels_headers, idx = nodes_labels_mega(index_to_text_file)

        rank_table['cortex_part'] =  labels_headers
        rank_table['mutual'] = (rank_table['weighted_ranks']+rank_table['non-weighted_ranks'])
        rank_table['mutual_rank'] = rank_table['mutual'].rank().astype('int64')

        #rank_table.sort_values("mutual_rank", inplace = True)
        table_file_name = folder_name + r'\node_degree_rank.csv'
        rank_table.to_csv(table_file_name)
        nodes_nw[i,:] = np.asarray(rank_table['non-weighted_ranks'])
        nodes_w[i,:] = np.asarray(rank_table['weighted_ranks'])

    nodes_nw = np.asarray(nodes_nw)
    nodes_w = np.asarray(nodes_w)
    np.save(r'C:\Users\Admin\my_scripts\Ax3D_Pack\Testings\nodedeg_nw',nodes_nw)
    np.save(r'C:\Users\Admin\my_scripts\Ax3D_Pack\Testings\nodedeg_w',nodes_w)
'''
    plt.figure(figsize=[15,15])
    ax0 = plt.subplot(2,1,1)
    ax0.set_title('Node degree distribution (Number of fibers)',fontsize=25)
    ax0.hist(nodes_nw[nodes_nw>0], bins = 100)
    ax0.set_ylabel('Frequency',fontsize=15)
    ax0.set_xlabel('Node degree level',fontsize=15)


    ax1 = plt.subplot(2,1,2)
    ax1.set_title('Node degree distribution (AxCaliber)', fontsize=25)
    ax1.hist(nodes_w[nodes_w>0], bins = 100)
    ax1.set_ylabel('Frequency',fontsize=15)
    ax1.set_xlabel('Node degree level',fontsize=15)

    plt.show()
'''





