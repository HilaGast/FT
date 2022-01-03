import os
import nibabel as nib
import numpy as np

def load_ft(tract_path, nii_ref):
    from dipy.io.streamline import load_trk, load_tck, Space

    if tract_path.endswith('.trk'):
        streams = load_trk(tract_path, nii_ref, Space.RASMM)
    elif tract_path.endswith('.tck'):
        streams = load_tck(tract_path, nii_ref, Space.RASMM)
    else:
        print("Couldn't recognize streamline file type")

    streamlines = streams.get_streamlines_copy()

    return streamlines


def load_weight_by_img(folder_name, weight_by):
    for file in os.listdir(folder_name):
        if weight_by in file and file.endswith(weight_by+'.nii') and not file.startswith("r"):
            weight_by_file = os.path.join(folder_name,file)
            continue
    weight_by_img = nib.load(weight_by_file)
    weight_by_data = weight_by_img.get_fdata()
    affine = weight_by_img.affine

    return weight_by_data, affine


def load_nii_file(diff_file_name):

    hardi_img = nib.load(diff_file_name)
    data = hardi_img.get_fdata()
    data[data<0]=0
    affine = hardi_img.affine

    return data, affine


def load_pve_files(folder_name,pve_file_name = '',tissue_labels_file_name=''):

    f_pve_csf = ''
    f_pve_gm = ''
    f_pve_wm = ''
    three_tissue_labels = ''
    for file in os.listdir(folder_name):
        if file.endswith(f"{pve_file_name}_0.nii") or file.endswith(f"CSF_{pve_file_name}"):
            f_pve_csf = os.path.join(folder_name, file)
        if file.endswith(f"{pve_file_name}_1.nii") or file.endswith(f"GM_{pve_file_name}"):
            f_pve_gm = os.path.join(folder_name, file)
        if file.endswith(f"{pve_file_name}_2.nii") or file.endswith(f"WM_{pve_file_name}"):
            f_pve_wm = os.path.join(folder_name, file)
        if file.endswith(tissue_labels_file_name):
            three_tissue_labels = os.path.join(folder_name, file)


    return f_pve_csf, f_pve_gm, f_pve_wm, three_tissue_labels


def bval_bvec_2_gtab(folder_name, small_delta=15.5):
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table

    for file in os.listdir(folder_name):
        if file.endswith(".bvec"):
            bvec_file = os.path.join(folder_name, file)
    bval_file = bvec_file[:-4:]+'bval'
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    bvals = np.around(bvals, decimals=-2)
    gtab = gradient_table(bvals, bvecs, small_delta=small_delta)

    return gtab


def load_mask(folder_name, mask_type):
    file_list = os.listdir(folder_name)
    for file in file_list:
        if 'mask' in file and mask_type in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            mask_mat = mask_img.get_data()
    return mask_mat