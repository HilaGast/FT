import numpy as np


def multi_comp_correction(r, p):
    from statsmodels.stats.multitest import multipletests as fdr
    import copy
    print('Correction for multiple comparisons')
    r = np.asarray(r)
    p = np.asarray(p)
    for_comp = [p > 0]
    p_corr_fc = fdr(p[for_comp], 0.05, 'fdr_bh')[1]
    p_corr = p
    p_corr[for_comp] = p_corr_fc

    r_th = np.asarray(copy.deepcopy(r))
    r_th[np.asarray(p_corr) > 0.05] = 0

    return list(r),list(p),list(r_th)


def calc_corr(x,y,fdr_correct = False, remove_outliers = False):
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
    from draw_scatter_fit import remove_outliers_y,remove_nans

    if len(np.shape(y))==2:
        r_vec = []
        p_vec = []
        for i in range(np.shape(y)[1]):
            yi = y[:,i]
            xi,yi = remove_nans(x,yi)

            if remove_outliers:
                xi,yi = remove_outliers_y(xi,yi)

            r, p = pearsonr(xi, yi)

            r_vec.append(r)
            p_vec.append(p)

        if fdr_correct:
            r_vec, p_vec, r_th_vec = multi_comp_correction(r_vec,p_vec)
            return r_th_vec, p_vec

        else:
            r_vec = np.asarray(r_vec)
            r_vec[np.asarray(p_vec)>0.05] = 0
            return list(r_vec), list(p_vec)

    else:
        x,y = remove_nans(x,y)

        if remove_outliers:
            x,y = remove_outliers_y(x,y)

        r, p = pearsonr(x,y)
        if p>0.05:
            r=0

        return r, p




def calc_corr_mat(x,y,fdr_correct = True, remove_outliers = False):

    from scipy.stats import pearsonr
    from draw_scatter_fit import remove_outliers_y,remove_nans

    r_vec = []
    p_vec = []
    for i in range(np.shape(y)[1]):
        yi = y[:,i]
        xi = x[:,i]
        xi,yi = remove_nans(xi,yi)

        if remove_outliers:
            xi,yi = remove_outliers_y(xi,yi)

        r, p = pearsonr(xi, yi)

        r_vec.append(r)
        p_vec.append(p)

    if fdr_correct:
        r_vec, p_vec, r_th_vec = multi_comp_correction(r_vec,p_vec)
        return r_th_vec, p_vec

    else:
        r_vec = np.asarray(r_vec)
        r_vec[np.asarray(p_vec)>0.05] = 0
        return list(r_vec), list(p_vec)

        return r, p

