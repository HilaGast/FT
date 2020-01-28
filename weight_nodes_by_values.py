import nibabel as nib
import numpy as np


def load_labels_nii_file(labels_file_name):
    nii_file = labels_file_name
    labels_img = nib.load(nii_file)

    return labels_img


def load_node_weights(folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\Testings'):

    #mat = np.load(folder_name+r'\clus_w.npy')
    mat = np.load(folder_name+r'\nodedeg_nw.npy')


    return mat


def weight_labels(labels_img, w_mat):
    m = np.median(w_mat,axis=0)

    weights = labels_img.get_data()
    for i in range(len(m)):
        weights[weights==i+1]=m[i]

    new_nii = nib.Nifti1Image(weights, labels_img.affine, labels_img.header)

    return new_nii


def save_weighted_nii_nodes(new_nii, folder_name):


    nib.save(new_nii,folder_name+r'\nw_node_deg.nii')


if __name__ == '__main__':
    labels_file_name = r'C:\Users\Admin\my_scripts\aal\megaatlas\MegaAtlas_Labels_highres.nii'
    labels_img = load_labels_nii_file(labels_file_name)
    folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\Testings'
    w_mat = load_node_weights(folder_name)
    new_nii = weight_labels(labels_img, w_mat)
    save_weighted_nii_nodes(new_nii, folder_name)