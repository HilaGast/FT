import os,glob
import numpy as np
import nibabel as nib

def norm_mat(subj_folder, cm_name, atlas):
    cm = np.load(cm_name)
    if atlas == 'bna':
        lookup = np.load(glob.glob(fr'{subj_folder}cm{os.sep}bna*lookup.npy')[0])
        parc = fr"{subj_folder}rBN_Atlas_274_combined_1mm.nii"
        parc_data = nib.load(parc).get_fdata()

    areas, count = np.unique(parc_data, return_counts=True)
    areas = list(np.asarray(areas[1:])-1)
    count = count[1:]

    lab_sizes = {a: c for a, c in zip(areas, count)}
    m = np.zeros((len(lookup), len(lookup)))


    for i, a1 in enumerate(lookup):
        for j, a2 in enumerate(lookup):
            try:
                mean_area = np.mean([lab_sizes[a1], lab_sizes[a2]])
                m[i, j] = mean_area
                m[j, i] = mean_area
            except KeyError:
                m[i, j] = 0
                m[j, i] = 0

    new_mat = cm * (m/np.nanmax(m))
    new_mat[np.isnan(new_mat)] = 0

    #import matplotlib.pyplot as plt
    #n_trk = np.sum(new_mat, 0)
    #n_trk = n_trk[lookup]
    #zero_size = np.sum(m,axis=0)<1
    #s_trk = [n_trk[i] for i,ok in enumerate(zero_size) if not ok]
    #plt.scatter(s_trk,count)
    #plt.show()


    np.save(fr'{cm_name[:-4]}_corrected.npy', new_mat)

    return new_mat


