from weighted_tracts import create_seeds, create_csd_model, create_fa_classifier, create_streamlines, save_ft
import os
from dipy.tracking import utils
import numpy as np
import nibabel as nib
from dipy.align.reslice import reslice


def load_dwi_files_blinds(folder_name):
    from dipy.core.gradients import gradient_table
    from dipy.core.geometry import compose_matrix, decompose_matrix

    for file in os.listdir(folder_name):
        if file.endswith(".bvec"):
            bvec_file = os.path.join(folder_name, file)
        if file.endswith("labels.nii"):
            labels_file_name = os.path.join(folder_name, file)
        if file.endswith("WM.nii"):
            wm_file_name = os.path.join(folder_name, file)
    bval_file = bvec_file[:-4:]+'bval'
    nii_file = bvec_file[:-4:]+'nii'
    hardi_img = nib.load(nii_file)
    data = hardi_img.get_fdata()
    affine = hardi_img.affine
    scale, shear, ang, trans, pre = decompose_matrix(affine)
    shear = np.zeros(np.shape(shear))
    affine = compose_matrix(scale, shear, ang, trans, pre)
    gtab = gradient_table(bval_file, bvec_file)
    #voxel_size = nib.affines.voxel_sizes(affine)
    #data, affine1 = reslice(data, affine, voxel_size, (3., 3., 3.))

    labels_img = nib.load(labels_file_name)
    labels = labels_img.get_fdata()
    #labels,affine1=reslice(labels,affine,voxel_size,(3., 3., 3.))

    wm_img = nib.load(wm_file_name)
    white_matter = wm_img.get_fdata()
    #white_matter,affine1=reslice(white_matter,affine,voxel_size,(3., 3., 3.))

    return gtab,data,affine,labels,white_matter,nii_file,bvec_file


def nodes_labels(lab_labels):

    lab_labels_index = [labels for labels in lab_labels]
    lab_labels_index = np.asarray(lab_labels_index, dtype='int')

    return lab_labels_index


def comp_con_mat(n,fa, streamlines,lab_labels_index,affine, folder_name):
    import scipy.io as sio
    from dipy.tracking.streamline import values_from_volume

    num, grouping = utils.connectivity_matrix(streamlines, affine, lab_labels_index,
                                                    return_mapping=True,
                                                    mapping_as_streamlines=True)
    num_mat = num[1:,1:]
    num_mat = np.asarray(num_mat,dtype='float64')
    vol_vec = fa.flatten()
    q = np.quantile(vol_vec[vol_vec>0], 0.95)
    fa_mat = np.zeros(np.shape(num_mat), dtype='float64')

    for pair, tracts in grouping.items():
        if pair[0] == 0 or pair[1] == 0:
            continue
        else:
            mean_vol_per_tract = []
            vol_per_tract = values_from_volume(fa, tracts, affine=affine)
            for s in vol_per_tract:
                s = np.asanyarray(s)
                non_out = [s < q]
                mean_vol_per_tract.append(np.nanmean(s[non_out]))

            mean_path_vol = np.nanmean(mean_vol_per_tract)

            fa_mat[pair[0]-1, pair[1]-1] = mean_path_vol
            fa_mat[pair[1]-1, pair[0]-1] = mean_path_vol

    mat_file_name = rf'{folder_name}\{n}_con_mat.mat'
    sio.savemat(mat_file_name, {'number_of_tracts': num_mat,'fa':fa_mat})


if __name__ == '__main__':
    subj_folder = r'C:\Users\HilaG\Desktop\blind dti\ctrl'
    all_folders = os.listdir(subj_folder)

    for s in all_folders[::]:
        name = s
        name = '/' + name
        n = name.replace('/', os.sep)
        folder_name = subj_folder + n
        dir_name = folder_name + '\streamlines'
        gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files_blinds(folder_name)
        csd_fit = create_csd_model(data, gtab, white_matter, sh_order=6)
        fa, classifier = create_fa_classifier(gtab, data, white_matter)
        lab_labels_index = nodes_labels(labels)
        seeds = create_seeds(folder_name, lab_labels_index, affine, use_mask=False, den=4)
        streamlines = create_streamlines(csd_fit, classifier, seeds, affine)
        save_ft(folder_name, n, streamlines, nii_file, file_name="_wholebrain_5d_labelmask.trk")
        comp_con_mat(n, fa, streamlines, lab_labels_index, affine, folder_name)
