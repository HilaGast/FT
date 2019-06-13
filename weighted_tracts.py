import os
import fnmatch
import nibabel as nib
from dipy.tracking import utils
from dipy.core.gradients import gradient_table
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
import numpy as np


def load_dwi_files(folder_name, small_delta=15.5):
    '''
    param:
        folder_name -
        small_delta -
    return:
        white_matter - the entire brain mask to track fibers
    '''

    for file in os.listdir(folder_name):
        if file.endswith(".bvec"):
            bvec_file = os.path.join(folder_name, file)
        if file.endswith("brain_seg.nii"):
            labels_file_name = os.path.join(folder_name,file)
    bval_file = bvec_file[:-4:]+'bval'
    nii_file = bvec_file[:-4:]+'nii'
    hardi_img = nib.load(nii_file)
    data = hardi_img.get_data()
    affine = hardi_img.affine
    gtab = gradient_table(bval_file, bvec_file, small_delta=small_delta)
    labels_img = nib.load(labels_file_name)
    labels = labels_img.get_data()
    white_matter = (labels == 3) | (labels == 2)  # 3-WM, 2-GM

    return gtab,data,affine,labels,white_matter,nii_file


def load_mask(folder_name, mask_type):
    file_list = os.listdir(folder_name)
    for file in file_list:
        if fnmatch.fnmatch(file,'mask') and fnmatch.fnmatch(file, mask_type) and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
    mask_img = nib.load(mask_file)
    mask_mat = mask_img.get_data()
    return mask_mat


def create_seeds(folder_name, white_matter, affine, use_mask = False, mask_type='cc'):
    if use_mask:
        mask_mat = load_mask(folder_name,mask_type)
        seed_mask = mask_mat == 1
    else:
        seed_mask = white_matter
    seeds = utils.seeds_from_mask(seed_mask, density=3, affine=affine)
    return seeds


def create_csd_model(data, gtab, white_matter, sh_order=4):
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

    csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order)
    csd_fit = csd_model.fit(data, mask=white_matter)

    return csd_fit


def create_fa_classifier(gtab,data,white_matter):
    import dipy.reconst.dti as dti
    from dipy.reconst.dti import fractional_anisotropy

    tensor_model = dti.TensorModel(gtab)
    tenfit = tensor_model.fit(data, mask=white_matter)
    fa = fractional_anisotropy(tenfit.evals)
    classifier = ThresholdTissueClassifier(fa, .2)

    return fa, classifier


def create_streamlines(csd_fit, classifier, seeds, affine):
    from dipy.data import default_sphere
    from dipy.direction import DeterministicMaximumDirectionGetter
    from dipy.tracking.streamline import Streamlines

    detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                                 max_angle=40.,
                                                                 sphere=default_sphere)

    streamlines = Streamlines(LocalTracking(detmax_dg, classifier, seeds, affine, step_size=.1))

    long_streamlines = np.ones((len(streamlines)), bool)
    for i in range(0, len(streamlines)):
        if streamlines[i].shape[0] < 70:
            long_streamlines[i] = False
    streamlines = streamlines[long_streamlines]

    return streamlines


def weighting_streamlines(streamlines,nii_file, weight_by = 'pasiS',hue = [0.0,1.0],saturation = [0.0,1.0],scale = [0,9]):
    from dipy.tracking.streamline import transform_streamlines
    weight_by_file = nii_file[:-3:]+'_'+weight_by+'.nii'
    weight_by_img = nib.load(weight_by_file)
    weight_by = weight_by_img.get_file()
    affine = weight_by_img.get_affine()
    streamlines_native = transform_streamlines(streamlines, np.linalg.inv(affine))
    lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                           saturation_range=saturation, scale_range=scale)
    streamlines_actor = actor.line(streamlines_native, weight_by, linewidth=0.1, lookup_colormap=lut_cmap)
    bar = actor.scalar_bar(lut_cmap)

    r = window.Renderer()
    r.add(streamlines_actor)
    r.add(bar)
    window.record(r, out_path='bundle2.png', size=(800, 800))
    window.show(r)
    #return streamlines, streamlines_native,


if __name__ == '__main__':
    folder_name = ''
    mask_type = 'cc'
    gtab, data, affine, labels, white_matter, nii_file = load_dwi_files(folder_name)
    