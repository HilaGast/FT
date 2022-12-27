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

def remove_outliers_Cooks(x,y,maxi=5):
    import statsmodels.api as sm
    np.set_printoptions(suppress=True)
    orig_len = len(y)
    removei = len(y)
    round=0
    while removei>maxi:
        round+=1
        rmodel = sm.OLS(y, x).fit()
        influence = rmodel.get_influence()
        cooks = influence.cooks_distance
        ii = np.where(cooks[0] <= (4 / len(y)))
        x = list(np.asarray(x)[ii])
        y = list(np.asarray(y)[ii])
        removei = orig_len-len(ii[0])
        print(f' Removed {str(removei)} outliers in round {str(round)}')
        orig_len = len(ii[0])

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



def draw_scatter_fit(x,y,fit_type = 'poly', deg=1, remove_outliers = True, ttl=None, comp_reg = False, norm_x = True):
    txt1 = None
    x,y = remove_nans(x,y)
    if remove_outliers:
        x,y = remove_outliers_Cooks(x,y)
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
            #txt1 = f'r({str(num)}) = {str(np.round(r,2))}, p = {str(np.round(p,3))}'
            txt1 = f'r({str(num)}) = {str(np.round(r,2))}, p = {p:.2e}'
    plt.style.use('dark_background')
    fig,ax = plt.subplots()
    ax.plot(x,y,'og',t,pol(t),'-w', markersize = 2)
    ax.tick_params(labelsize=14)
    if ttl:
        plt.title(ttl, {'fontsize':14})
    if txt1:
        fig.text(0.5,0.14,txt1,fontsize=14, bbox=dict(facecolor=[0.4, 0.7,0.6], alpha=0.6))
    print(f'R^2 = {str(r2)}')
    fig.tight_layout()
    plt.show()