import glob,os
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.cm as cm

subj_list = glob.glob(f'G:\data\V7\HCP\*{os.sep}')

add_mat=[]
fa_mat=[]
i=0
for sl in subj_list[:52]:
    try:
        add_cm = np.load(f'{sl}cm{os.sep}add_mega_unsifted_cm_ord.npy')
        #add_cm = np.load(f'{sl}cm{os.sep}add_mega_cm_ord.npy')
        add_mat.append(add_cm)

        fa_cm = np.load(f'{sl}cm{os.sep}fa_mega_unsifted_cm_ord.npy')
        #fa_cm = np.load(f'{sl}cm{os.sep}fa_mega_cm_ord.npy')
        fa_mat.append(fa_cm)
        i+=1
    except FileNotFoundError:
        continue
print(i)
add = np.asarray(add_mat)
add = add/np.max(add)
fa = np.asarray(fa_mat)
fa = fa/np.max(fa)

mask_mat = add > 0
mask_mat = np.sum(mask_mat,axis=0)/mask_mat.shape[0]
mask_mat = mask_mat>0.1 #choose only edges that appeared in >x% of subjects

r_mat = np.zeros((add.shape[1],add.shape[1]))
p_mat = np.zeros((add.shape[1],add.shape[1]))
for row in range(add.shape[1]):
    for col in range(row+1):
        if mask_mat[row,col]:
            fa_vec = fa[:,row,col]
            add_vec = add[:,row,col]
            na_vals = ~np.logical_or(np.isnan(fa_vec), np.isnan(add_vec))
            zero_vals = ~np.logical_or(fa_vec==0,add_vec==0)
            remove_ind = np.logical_and(na_vals,zero_vals)
            x = np.compress(remove_ind, fa_vec)
            y = np.compress(remove_ind, add_vec)

            r, p = pearsonr(x,y)
            r_mat[row,col] = r
            r_mat[col,row] = r
            p_mat[row,col] = p
            p_mat[col,row] = p
        else:
            r_mat[row,col] = 0
            r_mat[col,row] = 0
            p_mat[row,col] = 1
            p_mat[col,row] = 1


mat_title = 'AxCaliber-FA Correlation - all fibers (Pearson r)'
plt.figure(1, [40, 30])
cmap = cm.seismic
plt.imshow(r_mat, interpolation='nearest', cmap=cmap, origin='upper',vmax=1,vmin=-1)
plt.colorbar()
plt.xticks(ticks=np.arange(0, len(r_mat), 1), labels=[])
plt.yticks(ticks=np.arange(0, len(r_mat), 1), labels=[])
plt.title(mat_title, fontsize=44)
plt.tick_params(axis='x', pad=12.0, labelrotation=90, labelsize=12)
plt.tick_params(axis='y', pad=12.0, labelsize=12)

np.save(r'C:\Users\HilaG\Desktop\AxCaliber_FA_r_no_th_all_unsifted',r_mat)
plt.savefig(r'C:\Users\HilaG\Desktop\AxCaliber_FA correlation matrix (all fibers, unsifted - no th).png')
plt.show()