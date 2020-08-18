from weighted_tracts import *
from scipy.stats import pearsonr
subj = all_subj_folders
names = all_subj_names
labels_headers, idx = nodes_labels_mega(index_to_text_file)
fa_all = np.zeros((len(labels_headers), len(labels_headers), len(subj)))
axcaliber_all = np.zeros((len(labels_headers), len(labels_headers), len(subj)))
fa_mat_name = 'weighted_mega_wholebrain_4d_labmask_FA_nonnorm'
dti_fa_mat_name = 'weighted_mega_wholebrain_4d_labmask_FA_DTI_nonnorm'
axcaliber_mat_name = 'weighted_mega_wholebrain_4d_labmask_nonnorm'
for i, (s, n) in enumerate(zip(subj, names)):
    folder_name = subj_folder + s
    if f'{fa_mat_name}.npy' not in os.listdir(folder_name):
        print('No matching FA file')
        continue
    print(n)
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
    mat_fa = np.load(f'{folder_name}\{fa_mat_name}.npy')
    mat_axcaliber = np.load(f'{folder_name}\{axcaliber_mat_name}.npy')
    fa_all[:,:,i] = mat_fa/100
    axcaliber_all[:,:,i] = mat_axcaliber
count=0
r_mat = np.zeros((len(labels_headers),len(labels_headers)))
p_mat = np.zeros((len(labels_headers),len(labels_headers)))
for row in range(len(labels_headers)):
    for col in range(row+1):
        rc_fa = fa_all[row,col,:]
        rc_axcaliber = axcaliber_all[row,col,:]
        na_vals = ~np.logical_or(np.isnan(rc_fa), np.isnan(rc_axcaliber))
        zero_vals = ~np.logical_or(rc_fa==0,rc_axcaliber==0)
        remove_ind = np.logical_and(na_vals,zero_vals)
        x = np.compress(remove_ind, rc_fa)
        y = np.compress(remove_ind, rc_axcaliber)
        if len(x)<2 or len(y)<2:
            r_mat[row, col] = 0
            r_mat[col, row] = 0
            p_mat[row, col] = 1
            p_mat[col, row] = 1
            count+=1
            continue
        r, p = pearsonr(x,y)
        r_mat[row,col] = r
        r_mat[col,row] = r
        p_mat[row,col] = p
        p_mat[col,row] = p
        #plt.scatter(x,y)
        #plt.show()

#r_mat[p_mat>=0.05]=0
r_mat[np.isnan(r_mat)] = 0
#draw_r_mat:

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

mat_title = 'AxCaliber-FA Correlation (Pearson r)'
plt.figure(1, [40, 30])
cmap = cm.seismic
plt.imshow(r_mat, interpolation='nearest', cmap=cmap, origin='upper',vmax=1,vmin=-1)
plt.colorbar()
plt.xticks(ticks=np.arange(0, len(r_mat), 1), labels=labels_headers)
plt.yticks(ticks=np.arange(0, len(r_mat), 1), labels=labels_headers)
plt.title(mat_title, fontsize=44)
plt.tick_params(axis='x', pad=12.0, labelrotation=90, labelsize=12)
plt.tick_params(axis='y', pad=12.0, labelsize=12)
    #plt.savefig(fig_name)
np.save(r'C:\Users\HilaG\Desktop\AxCaliber_FA_r_no_th',r_mat)
#plt.savefig(r'C:\Users\HilaG\Desktop\AxCaliber_FA correlation matrix (no th).png')
plt.show()
