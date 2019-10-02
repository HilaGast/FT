from FT.weighted_tracts import load_ft, nodes_labels_mega, nodes_by_index_mega
import matplotlib.pyplot as plt
from FT.all_subj import all_subj_names
from dipy.tracking import utils
import numpy as np
from dipy.tracking.streamline import values_from_volume
import nibabel as nib
import os

index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlas2nii.txt'
subj = all_subj_names
weight_by='pasiS'

for s in subj:
    folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5' + s
    tract_path = folder_name+ r'\streamlines' + s + '_wholebrain.trk'
    streamlines = load_ft(tract_path)
    for file in os.listdir(folder_name):
        if file.endswith(".bvec"):
            bvec_file = os.path.join(folder_name, file)
    nii_file = bvec_file[:-4:]+'nii'

    lab_labels_index, affine = nodes_by_index_mega(folder_name)
    labels_headers, idx = nodes_labels_mega(index_to_text_file)
    m, grouping = utils.connectivity_matrix(streamlines, lab_labels_index, affine=affine,
                                            return_mapping=True,
                                            mapping_as_streamlines=True)
    mm = m[1:]
    mm = mm[:,1:]
    mm = mm[idx]
    mm = mm[:, idx]

    weight_by_file = nii_file[:-4:] + '_' + weight_by + '.nii'
    weight_by_img = nib.load(weight_by_file)
    weight_by_data = weight_by_img.get_data()
    affine = weight_by_img.affine
    m_weighted = np.zeros((len(idx),len(idx)), dtype='float64')
    for pair, tracts in grouping.items():
        if pair[0] == 0 or pair[1] == 0:
            continue
        else:
            mean_vol_per_tract = []
            vol_per_tract = values_from_volume(weight_by_data, tracts, affine=affine)
            for s in vol_per_tract:
                mean_vol_per_tract.append(np.mean(s))
            mean_path_vol = np.nanmean(mean_vol_per_tract)
            m_weighted[pair[0]-1, pair[1]-1] = mean_path_vol
            m_weighted[pair[1]-1, pair[0]-1] = mean_path_vol

    mm_weighted = m_weighted[idx]
    mm_weighted = mm_weighted[:, idx]

    np.save(folder_name + r'\non-weighted_non-norm', mm)
    np.save(folder_name + r'\weighted_non-norm', mm_weighted)
    nw = np.reshape(mm,(23409,))
    w = np.reshape(mm_weighted, (23409,))

    plt.figure(figsize=[12,6])

    ax0 = plt.subplot(1,2,1)
    ax0.set_title('# tracts')
    ax0.hist(nw[nw>0], bins=30)


    ax1 = plt.subplot(1,2,2)
    ax1.set_title('Average Pasi')
    ax1.hist(w[w>0], bins=30)

    plt.show()
