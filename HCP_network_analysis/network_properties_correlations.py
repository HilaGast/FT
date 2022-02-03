import networkx as nx
import os, glob
import numpy as np
from network_analysis.global_network_properties import get_efficiency
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from draw_scatter_fit import remove_outliers_Cooks, remove_nans
from network_analysis.norm_cm_by_atlas_areas import norm_mat

subj_list = glob.glob(f'F:\data\V7\HCP\*{os.sep}')
add_eff=[]
num_eff=[]
add_cc=[]
num_cc=[]
all_add_mat = []
all_num_mat = []
cerbellum_i = [i for i in range(123,151)]

for sl in subj_list:

    add_cm = np.load(f'{sl}cm{os.sep}add_bna_cm_ord.npy')
    add_cm = np.delete(add_cm,cerbellum_i, axis=0)
    add_cm = np.delete(add_cm,cerbellum_i, axis=1)
    add_eff.append(get_efficiency(add_cm))
    G = nx.from_numpy_array(add_cm)
    cc_val = nx.clustering(G, weight='weight')
    add_cc.append(np.nanmean(list(cc_val.values())))
    all_add_mat.append(add_cm)

    try:
        num_cm = np.load(f'{sl}cm{os.sep}num_bna_cm_ord_corrected.npy')
    except FileNotFoundError:
        num_cm = norm_mat(sl,f'{sl}cm{os.sep}num_bna_cm_ord.npy','bna')

    #num_cm = np.load(f'{sl}cm{os.sep}num_bna_cm_ord.npy')
    num_cm = np.delete(num_cm,cerbellum_i, axis=0)
    num_cm = np.delete(num_cm,cerbellum_i, axis=1)
    num_eff.append(get_efficiency(num_cm))
    G = nx.from_numpy_array(num_cm)
    cc_val = nx.clustering(G, weight='weight')
    num_cc.append(np.nanmean(list(cc_val.values())))
    all_num_mat.append(num_cm)


all_add_mat = np.asarray(all_add_mat)
all_num_mat = np.asarray(all_num_mat)
mask_mat = all_num_mat > 0
mask_mat = np.sum(mask_mat,axis=0)/mask_mat.shape[0]
mask_mat = mask_mat>0.75 #choose only edges that appeared in >75% of subjects
add = all_add_mat.sum(0)/(all_add_mat != 0).sum(0) #mean of nonzeros
num = all_num_mat.sum(0)/(all_num_mat != 0).sum(0) #mean of nonzeros
masked_add = add * mask_mat
masked_num = num * mask_mat
masked_add[np.isnan(masked_add)] = 0
masked_num[np.isnan(masked_num)] = 0

masked_add = masked_add/np.nanmax(masked_add)
masked_num = masked_num/np.nanmax(masked_num)

masked_add[masked_add==0] = np.nan
masked_num[masked_num==0] = np.nan

add_nd = np.nanmean(masked_add, axis=0)
num_nd = np.nanmean(masked_num, axis=0)

#add_nd[np.isnan(add_nd)] = 0
#num_nd[np.isnan(num_nd)] = 0


fig,axs = plt.subplots(1,3,figsize=[15,5])
fig.tight_layout()
#Eglob:
x,y = remove_nans(num_eff, add_eff)
x,y = remove_outliers_Cooks(x,y,maxi=10)
axs[0].scatter(x,y,marker='.',color='k')
axs[0].set_title('Eglob')
axs[0].set_xlabel('NOS')
axs[0].set_ylabel('ADD')
r,p = pearsonr(x,y)
axs[0].text(1,1,f'r = {np.round(r,2)}, p = {np.round(p,2)}',fontsize=9, horizontalalignment = 'right',verticalalignment = 'bottom',in_layout=True)

#Clustering Coefficient:
x,y = remove_nans(num_cc,add_cc)
x,y = remove_outliers_Cooks(x,y,maxi=10)
axs[1].scatter(x,y,marker='.',color='k')
axs[1].set_title('Clustering Coefficient')
axs[1].set_xlabel('NOS')

r,p = pearsonr(x,y)
axs[1].text(0.3,0.8,f'r = {np.round(r,2)}, p = {np.round(p,2)}',fontsize=9, horizontalalignment = 'right',verticalalignment = 'bottom')

#Node Degree:
x,y = remove_nans(num_nd, add_nd)
x,y = remove_outliers_Cooks(x,y,maxi=10)
axs[2].scatter(x,y,marker='.',color='k')
axs[2].set_title('Average Node Degree')
axs[2].set_xlabel('NOS')

r,p = pearsonr(x,y)
axs[2].text(1,1,f'r = {np.round(r,2)}, p = {np.round(p,2)}',fontsize=9, horizontalalignment = 'right',verticalalignment = 'bottom')

plt.show()