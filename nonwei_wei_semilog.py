from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# load non weighted file:

nwei=np.load(r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5\DlYo_subj10\non-weighted_non-norm.npy')

# load weighted file:
wei=np.load(r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5\DlYo_subj10\weighted_non-norm.npy')


# all above 1 equals nan:
wei[wei>1]=np.inf
nwei[nwei>1]=np.inf


# start empty lists:
wei_h = list()
nwei_h = list()

# fill only once for each cortex area combination:
n=1
for r in wei:
    wei_h=wei_h+list(r[0:n])
    n+=1

n=1
for r in nwei:
    nwei_h=nwei_h+list(r[0:n])
    n+=1


# leave finite values only:
wei_h = np.asarray(wei_h)
nwei_h = np.asarray(nwei_h)

wei_r = wei_h[np.isfinite(wei_h)]
nwei_r = nwei_h[np.isfinite(nwei_h)]

# plot:
plt.semilogx(nwei_r,wei_r,'b.')
plt.xlabel('Num of tracts [log]')
plt.ylabel('Mean AxCaliber value of tracts')
plt.title('AxCaliber weighted tract values as a function of the number of tracts between GM areas',fontsize=10)
plt.show()

# add to a combined value:


# linear regression:
x = nwei_r
y = wei_r
x=x.reshape(-1,1)


model=LinearRegression()
model.fit(x,y)

# r^2:
rr = model.score(x,y)

# slope:
slope = model.coef_

# pval:
F, pval = f_regression(x,y)


wei_10 = wei_r
nwei_10 = nwei_r
