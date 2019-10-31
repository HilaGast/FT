import matplotlib.pyplot as plt
from FT.all_subj import all_subj_names,all_subj_folders
import numpy as np
import pandas as pd
from FT.weighted_tracts import nodes_labels_mega


def load_df(s):
    folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep' + s
    #table_file_name = folder_name + r'\node_degree_rank.csv'
    table_file_name = folder_name + r'\clustering_coeff_rank.csv'

    rank_table = pd.read_csv(table_file_name)

    return rank_table


def count_highest_ranking(count_ranks,n):
    a = rank_table.sort_values('non-weighted_ranks').tail(n).index
    count_ranks['Num_of_fibers'][a] += 1

    a = rank_table.sort_values('weighted_ranks').tail(n).index
    count_ranks['Axcaliber'][a] += 1

    #a = rank_table.sort_values('mutual_rank').tail(n).index
    #count_ranks['count_mutual'][a] += 1
    return count_ranks


def calc_percentage(count_ranks,n):
    count_ranks['Num_of_fibers']=count_ranks['Num_of_fibers'] / n * 100
    count_ranks['Axcaliber']=count_ranks['Axcaliber'] / n * 100
    #count_ranks['count_mutual']=count_ranks['count_mutual'] / n * 100

    return count_ranks


if __name__ == '__main__':
    subj = all_subj_folders.copy()
    index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlascortex2nii.txt'
    labels_headers, idx = nodes_labels_mega(index_to_text_file)
    data = np.zeros([len(subj),len(labels_headers)])
    #data = np.zeros([len(labels_headers), 4])
    #count_ranks = pd.DataFrame(data,columns=['cortex_part','count_numoffibers','count_axcaliber','count_mutual'],dtype='int64',index=idx)
    count_ranks = pd.DataFrame(data,columns=labels_headers,dtype='int64')

    count_num = count_ranks.copy()
    count_axc = count_ranks.copy()

    #count_ranks['cortex_part'] = labels_headers
    for i,s in enumerate(subj):
        rank_table = load_df(s)
        count_num.iloc[i]=np.asarray(rank_table['non-weighted_ranks'][:])
        count_axc.iloc[i]=np.asarray(rank_table['weighted_ranks'][:])
    num = count_num.mean(axis=0)
    axc = count_axc.mean(axis=0)
    df = pd.concat([num, axc], axis=1)

    plt.figure(figsize=[28,14])
    ax1 = plt.subplot(2,1,1)
    ax1.hist(df[0].sort_values)
    ax1.set_title('Mean rank of nodes (clustering-coefficient)\n Number of tracts', fontsize=25)
    ax1.set_xlabel('Cortex nodes', fontsize=15)
    ax1.set_ylabel('Mean rank', fontsize=15)
    ax1.set_xticks(idx)
    ax1.set_xticklabels(labels_headers)
    ax1.tick_params(axis='x', pad=8.0, labelrotation=90, labelsize=9)
    ax2 = plt.subplot(2,1,2)
    ax2.bar(idx,axc)
    ax2.set_title('Mean rank of nodes (clustering-coefficient)\n AxCaliber', fontsize=25)
    ax2.set_xlabel('Cortex nodes', fontsize=15)
    ax2.set_ylabel('Mean rank', fontsize=15)
    ax2.set_xticks(idx)
    ax2.set_xticklabels(labels_headers)
    ax2.tick_params(axis='x', pad=8.0, labelrotation=90, labelsize=9)
    plt.show()
    plt.scatter(num[0:50],num[50::])
    plt.show()
    plt.scatter(axc[0:50],axc[50::])
    plt.show()

    '''    
    count_ranks = calc_percentage(count_ranks,len(subj))

    condition1 = count_ranks.Num_of_fibers > 30
    condition2 = count_ranks.Axcaliber > 30
    to_show = condition1 | condition2

    ax = count_ranks[to_show].plot.bar(rot=0,figsize=[15,10],width=0.8)
    ax.set_title('Top-ranked nodes for high clutering-coefficient rank in either networks\n (inter-subject)',fontsize=25)
    ax.set_xlabel('Cortex nodes',fontsize=15)
    ax.set_ylabel('Frequency [%]',fontsize=15)
    ax.set_xticklabels(count_ranks[to_show].cortex_part)
    ax.tick_params(axis='x', pad=8.0, labelrotation=90, labelsize=11)

    plt.show()

    ax = count_ranks.plot.bar(rot=0,figsize=[15,10],width=0.8)
    ax.set_title('High clutering-coefficient rank in either networks\n (inter-subject)',fontsize=25)
    ax.set_xlabel('Cortex nodes',fontsize=15)
    ax.set_ylabel('Frequency [%]',fontsize=15)
    ax.set_xticklabels(count_ranks.cortex_part)
    ax.tick_params(axis='x', pad=8.0, labelrotation=90, labelsize=11)

    plt.show()
    ax = count_ranks.plot.line(figsize=[25,10])
    ax = count_ranks.plot.bar(rot=0,figsize=[25,10],width=0.8)'''

