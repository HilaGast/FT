import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def remove_outliers_IsoForest(x,y):
    from sklearn.ensemble import IsolationForest
    rng = np.random.RandomState(42)
    clf = IsolationForest(contamination=0.1, random_state=rng).fit(np.asarray([x,y]).T)
    find_out = clf.predict(np.asarray([x,y]).T)

    x = list(np.asarray(x)[find_out>0])
    y = list(np.asarray(y)[find_out>0])

    print(f' Removed {str(np.sum(find_out<0))} outliers')

    return x,y


def remove_outliers_y(x,y):
    my = np.nanmean(y)
    stdy = np.nanstd(y)
    find_out = np.ones(len(y))
    find_out[y>my+2.5*stdy] = -1
    find_out[y < my - 2.5 * stdy] = -1
    x = list(np.asarray(x)[find_out>0])
    y = list(np.asarray(y)[find_out>0])
    print(f' Removed {str(np.sum(find_out<0))} outliers')

    return x,y


def remove_nans(x,y):
    x = np.asarray(x)
    y=np.asarray(y)

    not_nans_x_pos = ~np.isnan(x)
    x = x[not_nans_x_pos]
    y = y[not_nans_x_pos]

    not_nans_y_pos = ~np.isnan(y)
    x = x[not_nans_y_pos]
    y=y[not_nans_y_pos]

    return list(x), list(y)



def draw_scatter_fit(x,y,fit_type = 'poly', deg=1, ttl=None, comp_reg = False, norm_x = True):
    txt1 = None
    x,y = remove_nans(x,y)
    x,y = remove_outliers_y(x,y)
    #x,y = remove_outliers_y(x,y)
    #y = list((np.asarray(y)-np.nanmin(y))/(np.nanmax(y)-np.nanmin(y)))
    if norm_x:
        x = list((np.asarray(x)-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x)))
    t = np.linspace(np.nanmin(x), np.nanmax(x), 100)

    if 'poly' == str.lower(fit_type):
        pol = np.poly1d(np.polyfit(x,y,deg))
        r2 = r2_score(y,pol(x))
        if deg == 1 and comp_reg:
            from scipy.stats import pearsonr
            r, p = pearsonr(x, y)
            num = np.sum(~np.isnan(y))
            txt1 = f'r({str(num)}) = {str(np.round(r,2))}, p = {str(np.round(p,3))}'
    fig,ax = plt.subplots()
    ax.plot(x,y,'og',t,pol(t),'-k', markersize = 4)
    if ttl:
        plt.title(ttl, {'fontsize':20})
    if txt1:
        fig.text(0.55,0.14,txt1,fontsize=11, bbox=dict(facecolor=[0.4, 0.7,0.6], alpha=0.2))
    print(f'R^2 = {str(r2)}')
    fig.tight_layout()
    plt.show()