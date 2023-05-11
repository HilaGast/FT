import os

import numpy as np


def calc_corr(x,y,trait_name, weight_by, figs_folder, show=True):
    '''

    :param x: list of values. len(x) = n
    :param y: values to compare. might be a list (len(y)=n) or a 2D matrix (y.shape[0]=n)
    :param fdr_correct: if y is a matrix and fdr_correct is True, fdr correction is performed to compute p
    :param remove_outliers: if remove_outliers is True, outliers from y is removed before correlation calculation
    :return:
    r: float64() or a list of pearson r stats
    p: float64() or a list of signficance
    '''

    from scipy.stats import pearsonr
    from draw_scatter_fit import remove_nans

    if len(np.shape(y))==2:
        r_vec = []
        p_vec = []
        for i in range(np.shape(y)[1]):
            yi = y[:,i]
            xi,yi = remove_nans(x,yi)

            r, p = pearsonr(xi, yi)
            r_vec.append(r)
            p_vec.append(p)


        r_vec = np.asarray(r_vec)
        r_vec[np.asarray(p_vec)>0.05] = 0
        return list(r_vec), list(p_vec)

    else:

        x,y = remove_nans(x,y)

        r, p = pearsonr(x,y)
        ttl = f'Predicted Vs Observed \n {trait_name} {weight_by} \n'
        fig_name = f'{figs_folder}{os.sep}Predicted vs Observed Pearson r {weight_by} {trait_name}.png'
        if show:
            draw_scatter_fit(x,y,r,p, ttl, fig_name)

        return r, p



def draw_scatter_fit(x, y, r, p, ttl, fig_name):
    '''
    :param x: list of values. len(x) = n
    :param y: values to compare. might be a list (len(y)=n) or a 2D matrix (y.shape[0]=n)
    :param r: float64() or a list of spearman r stats
    :param p: float64() or a list of signficance
    '''

    import matplotlib.pyplot as plt
    import numpy as np
    #txt1 = f'r = {str(np.round(r, 2))}, p = {p:.2e}'
    txt1 = f'r = {str(np.round(r, 2))}'
    print(f'r = {str(np.round(r, 2))}, p = {p:.2e}')
    fig,ax = plt.subplots()
    fig.set_size_inches(5, 5)
    pol = np.poly1d(np.polyfit(x, y, 1))
    if 'Num' in fig_name:
        dotcolor = [0.2, 0.7, 0.6]
    elif 'ADD' in fig_name:
        dotcolor = [0.8, 0.5, 0.3]
    elif 'FA' in fig_name:
        dotcolor = [0.3, 0.3, 0.5]
    else:
        dotcolor = 'r'
    plt.scatter(x, y, c=dotcolor, s=50, alpha=0.5)
    plt.plot(x,pol(x), 'k-', lw=3)
    fig.text(0.7, 0.18, txt1, fontsize=14, bbox=dict(facecolor=[0.4, 0.4, 0.4], alpha=0.4))
    ax.tick_params(labelsize=14)
    plt.xlabel('Test', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    plt.ylim([60, 160])
    plt.xlim([60, 160])
    plt.title(ttl, fontsize=15)
    fig.tight_layout()
    plt.savefig(fig_name)
    plt.show()

    return