import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from draw_scatter_fit import remove_nans, remove_outliers_y, remove_outliers_Cooks


def draw_scatter_fit_weights(x, y, xlabel, ylabel, xlim, ylim, c='r'):
    x, y = remove_nans(x, y)
    #x, y = remove_outliers_Cooks(x, y)
    t = np.linspace(xlim[0], xlim[1], 200)
    pol = np.poly1d(np.polyfit(x, y, 1))
    r, p = pearsonr(x, y)
    num = np.sum(~np.isnan(y))
    txt1 = f'r({str(num)}) = {str(np.round(r, 2))}, p = {p:.2e}'

    #plt.style.use('Solarize_Light2')
    fig, ax = plt.subplots()
    ax.plot(x, y, '.k', markersize=5, alpha=0.2)
    ax.plot(t, pol(t), '-',c=c, linewidth=5)
    ax.tick_params(labelsize=14)
    #ax.set_aspect('equal','box')
    ax.set_box_aspect(1)
    fig.text(0.3, 0.87, txt1, fontsize=14, bbox=dict(facecolor=c, alpha=0.5))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    add = np.load(r'G:\data\V7\HCP\cm\average_yeo7_200_ADD_HistMatch_SC_atleast_half_subjects.npy')
    fa = np.load(r'G:\data\V7\HCP\cm\average_yeo7_200_FA_HistMatch_SC_atleast_half_subjects.npy')
    nos = np.load(r'G:\data\V7\HCP\cm\average_yeo7_200_Num_HistMatch_SC_atleast_half_subjects.npy')
    dist = np.load(r'G:\data\V7\HCP\cm\yeo7_200_euclidean_dist.npy')

    add = np.triu(add, 0)
    fa = np.triu(fa, 0)
    nos = np.triu(nos, 0)
    dist = np.triu(dist, 0)
    add[add == 0] = np.nan
    fa[fa == 0] = np.nan
    nos[nos == 0] = np.nan
    dist[dist == 0] = np.nan
    log_nos = np.log(nos)
    log_add = np.log(add)
    log_fa = np.log(fa)
    # draw_scatter_fit_weights(nos, add, 'NOS', 'ADD', (1, 30), (3, 8))
    # draw_scatter_fit_weights(nos, fa, 'NOS', 'FA', (1, 30), (0.1, 0.7))
    # draw_scatter_fit_weights(add, fa, 'ADD', 'FA', (3, 8), (0.1, 0.7))

    # draw_scatter_fit_weights(nos, add, 'NOS', 'ADD', (1, 25), (1, 25))
    # draw_scatter_fit_weights(nos, fa, 'NOS', 'FA', (1, 25), (1, 25))
    # draw_scatter_fit_weights(add, fa, 'ADD', 'FA', (1, 25), (1, 25))

    #
    draw_scatter_fit_weights(dist, log_nos, 'Dist', 'log(NOS)', (1, 150), (0, 5), c=[0.2, 0.7, 0.6])
    draw_scatter_fit_weights(dist, log_add, 'Dist', 'log(ADD)', (0, 150), (0, 5), c=[0.8, 0.5, 0.3])
    draw_scatter_fit_weights(dist, log_fa, 'Dist', 'log(FA)', (0, 150), (0, 5), c=[0.3, 0.3, 0.5])


