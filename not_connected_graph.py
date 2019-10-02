import matplotlib.pyplot as plt
from FT.weighted_tracts import nodes_labels_mega
import numpy as np
from FT.all_subj import all_subj_names


index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlascortex2nii.txt'
labels_headers, idx = nodes_labels_mega(index_to_text_file)
who_has_no_con = np.zeros([len(all_subj_names), len(idx)])
for i, s in enumerate(all_subj_names):
    a = np.load(r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5' + s + r'\non-weighted_mega_wholebrain_cortex_nonnorm.npy')
    b = np.sum(a, axis=1)
    who_has_no_con[i, :] = b

x = who_has_no_con < 1
no_con = np.sum(x, axis=0)

plt.figure(1, [30, 15])
plt.tick_params(axis='x', pad=10, labelrotation=90, labelsize=11)
plt.bar(labels_headers,no_con)
plt.show()