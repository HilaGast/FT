
import numpy as np
import os
from os.path import join as pjoin
from scipy import stats
from statsmodels.stats.multitest import multipletests

def load_and_stack_matrices(folder_name,mat_name):
    subjs = os.listdir(folder_name)
    mat = np.load(pjoin(folder_name,subjs[0],mat_name))
    for s in subjs[1::]:
        mati = np.load(pjoin(folder_name,s,mat_name))
        mati = mati[:,:,np.newaxis]
        mat = np.dstack((mat,mati))

    return mat


def calc_ttest_mat(mat1,mat2,axis=2):
    tmat,pmat = stats.ttest_rel(mat1,mat2,axis)
    #p = np.tril(pmat)
    #pvec = p[p>0]
    #pcor = multipletests(pvec,alpha=0.1,method='fdr_bh')[1]
    #p[p>0] = pcor
    #pmat=p
    #tmat = np.tril(tmat)

    return tmat,pmat


def draw_stat_mat(mat,type):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from weighted_tracts import nodes_labels_yeo7, nodes_labels_aal3
    from all_subj import index_to_text_file

    labels_headers, idx = nodes_labels_aal3(index_to_text_file)

    mat_title = 'T-test values within subject (before/after learning balance task) - eyes opened'
    plt.figure(1, [40, 30])
    cmap = cm.seismic
    plt.imshow(mat, interpolation='nearest', cmap=cmap, origin='upper', vmax=5, vmin=-5)
    plt.colorbar()
    plt.xticks(ticks=np.arange(0, len(mat), 1), labels=labels_headers)
    plt.yticks(ticks=np.arange(0, len(mat), 1), labels=labels_headers)
    plt.title(mat_title, fontsize=44)
    plt.tick_params(axis='x', pad=12.0, labelrotation=90, labelsize=12)
    plt.tick_params(axis='y', pad=12.0, labelsize=12)
    # plt.savefig(fig_name)
    np.save(rf'F:\Hila\balance\eo_{type}_norm_num-add', mat)
    #plt.savefig(r'F:\Hila\balance\ec\pval.png')
    plt.show()



if __name__ == "__main__":
    mat_name = r'norm_num-add_mat.npy'
    folder_before = r'F:\Hila\balance\eo\before'
    folder_after = r'F:\Hila\balance\eo\after'
    mat_before = load_and_stack_matrices(folder_before,mat_name)
    mat_after = load_and_stack_matrices(folder_after,mat_name)
    tmat,pmat = calc_ttest_mat(mat_before,mat_after)
    tmat[abs(pmat)>0.05]=0
    draw_stat_mat(pmat,type='pval_aal')
    draw_stat_mat(tmat,type='ttest_aal')

