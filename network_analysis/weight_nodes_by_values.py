import nibabel as nib
import numpy as np
import scipy.io as sio


def load_labels_nii_file(labels_file_name):
    nii_file = labels_file_name
    labels_img = nib.load(nii_file)

    return labels_img


def load_node_weights(folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\Testings'):

    #mat = np.load(folder_name+r'\clus_w.npy')
    mat = np.load(folder_name+r'\nodedeg_nw.npy')


    return mat


def weight_labels(labels_img, c):
    m = np.zeros(170)
    m[0:34] = c[0:34]
    m[36:80] = c[34:78]
    m[82::] = c[78::]

    weights = labels_img.get_fdata()
    for i in range(len(m)):
        weights[weights==i+1]=m[i]

    weights = np.asarray(weights,dtype='int16')
    new_nii = nib.Nifti1Image(weights, labels_img.affine, labels_img.header)

    return new_nii


def save_weighted_nii_nodes(new_nii, folder_name):

    nib.save(new_nii,folder_name+r'\communities_fa.nii')


if __name__ == '__main__':
    labels_file_name = r'F:\Hila\Ax3D_Pack\Surface_visualization\AAL3_highres_atlas_corrected.nii'
    labels_img = load_labels_nii_file(labels_file_name)
    folder_name = r'C:\Users\HilaG\Desktop\4OlafSporns\surfaces\mean_over_50subj'
    #folder_name = r'C:\Users\HilaG\Desktop\4OlafSporns\surfaces\single_subj'

    #new_nii = weight_labels(labels_img, w_mat)
    #save_weighted_nii_nodes(new_nii, folder_name)
    weight_by='fa'
    communities_file = r'C:\Users\HilaG\Desktop\4OlafSporns\allsubjmatsbased\group_division_allsubj.mat'
    #communities_file = r'C:\Users\HilaG\Desktop\4OlafSporns\surfaces\single_subj\subj_communities1.mat'
    mat = sio.loadmat(communities_file)
    communities = np.asarray(mat['ciuv'])
    if weight_by == 'num':
        c = communities[:,0]
    elif weight_by == 'fa':
        c = communities[:, 1]
    elif weight_by == 'ax':
        c = communities[:, 2]

    new_nii = weight_labels(labels_img, c)
    save_weighted_nii_nodes(new_nii, folder_name)