from scipy.stats import ttest_ind, ttest_rel
import numpy as np
def ttest(x1,x2,type='independent', alternative='less'):
    '''
    :param x1: vec or matrix of values where last dim is different subjects
    :param x2: vec or matrix of values where last dim is different subjects
    :param type: ['independent', 'related']
    :param alternative: ['less', 'greater', 'two-sided']
    :return:
    '''
    vars_x1 = x1.shape
    vars_x2 = x2.shape
    if vars_x1[:-1] != vars_x2[:-1]:
        raise ValueError('x1 and x2 must have the same shape')

    if len(vars_x1) == 1:
        if type == 'independent':
            t, p = ttest_ind(x1, x2, alternative=alternative)
        elif type == 'related':
            t, p = ttest_rel(x1, x2, alternative=alternative)
        else:
            raise ValueError('type must be either independent or related')
        return t,p

    if len(vars_x1) == 2:
        t = np.zeros(vars_x1[0])
        p = np.zeros(vars_x1[0])
        for var in range(vars_x1[0]):
            x1_var = x1[var,:]
            x2_var = x2[var,:]
            if type == 'independent':
                t[var], p[var] = ttest_ind(x1_var, x2_var, alternative=alternative)
            elif type == 'related':
                t[var], p[var] = ttest_rel(x1_var, x2_var, alternative=alternative)
            else:
                raise ValueError('type must be either independent or related')
        return t,p

    if len(vars_x1) == 3:
        t = np.zeros((vars_x1[0], vars_x1[1]))
        p = np.zeros((vars_x1[0], vars_x1[1]))
        for var1 in range(vars_x1[0]):
            for var2 in range(vars_x1[1]):
                x1_var = x1[var1, var2, :]
                x2_var = x2[var1, var2, :]
                if type == 'independent':
                    t[var1, var2], p[var1, var2] = ttest_ind(x1_var, x2_var, alternative=alternative)
                elif type == 'related':
                    t[var1, var2], p[var1, var2] = ttest_rel(x1_var, x2_var, alternative=alternative)
                else:
                    raise ValueError('type must be either independent or related')
        return t,p