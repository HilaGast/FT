import nibabel as nib
from dipy.tracking import utils
from dipy.core.gradients import gradient_table
import numpy as np
from all_subj import *


def load_dwi_files(folder_name, small_delta=15.5):
    from dipy.io.gradients import read_bvals_bvecs
    '''
    param:
        folder_name -
        small_delta - 15 for thebase4ever, 15.5 for thebase
    return:
        white_matter - the entire brain mask to track fibers
    '''

    for file in os.listdir(folder_name):
        if file.endswith(".bvec"):
            bvec_file = os.path.join(folder_name, file)
        if file.endswith("brain_seg.nii"):
            labels_file_name = os.path.join(folder_name, file)

    bval_file = bvec_file[:-4:]+'bval'
    nii_file = os.path.join(folder_name,'diff_corrected.nii')
    hardi_img = nib.load(nii_file)
    data = hardi_img.get_fdata()
    data[data<0]=0
    affine = hardi_img.affine
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    bvals = np.around(bvals, decimals=-2)
    gtab = gradient_table(bvals, bvecs, small_delta=small_delta)
    labels_img = nib.load(labels_file_name)
    labels = labels_img.get_fdata()
    white_matter = (labels == 3) #| (labels == 2)  # 3-WM, 2-GM

    return gtab,data,affine,labels,white_matter,nii_file,bvec_file


def load_mask(folder_name, mask_type):
    file_list = os.listdir(folder_name)
    for file in file_list:
        if 'mask' in file and mask_type in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            mask_mat = mask_img.get_data()
    return mask_mat


def create_seeds(folder_name, lab_labels_index, affine, use_mask = True, mask_type='cc',den = 1):
    if use_mask:
        mask_mat = load_mask(folder_name,mask_type)
        seed_mask = mask_mat == 1
    else:
        seed_mask = lab_labels_index>0 #GM seeds
    seeds = utils.seeds_from_mask(seed_mask, density=den, affine=affine)
    return seeds


def create_csd_model(data, gtab, white_matter, sh_order=6):
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel,auto_response_ssst
    response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)

    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)
    csd_fit = csd_model.fit(data, mask=white_matter)

    return csd_fit


def create_fa_classifier(gtab,data,white_matter):
    import dipy.reconst.dti as dti
    from dipy.reconst.dti import fractional_anisotropy
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

    tensor_model = dti.TensorModel(gtab)
    tenfit = tensor_model.fit(data, mask=white_matter)
    fa = fractional_anisotropy(tenfit.evals)
    classifier = ThresholdStoppingCriterion(fa, .18)

    return fa, classifier


def create_streamlines(model_fit, seeds, affine, gtab=None, data=None, white_matter=None, classifier_type="fa"):
    from dipy.data import default_sphere
    from dipy.direction import DeterministicMaximumDirectionGetter
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking.local_tracking import LocalTracking

    detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(model_fit.shm_coeff,
                                                                 max_angle=30.,
                                                                 sphere=default_sphere)
    if classifier_type == "fa":
        print("Tractography using local tracking and FA clasifier")
        classifier = create_fa_classifier(gtab, data, white_matter)[1]
        print('Starting to compute streamlines')
        streamlines = Streamlines(LocalTracking(detmax_dg, classifier, seeds, affine, step_size=1,return_all=False))

    long_streamlines = np.ones((len(streamlines)), bool)
    for i in range(0, len(streamlines)):
        if streamlines[i].shape[0] < 100:
            long_streamlines[i] = False
    streamlines = streamlines[long_streamlines]

    return streamlines


def load_ft(tract_path, nii_file):
    from dipy.io.streamline import load_trk,Space

    streams = load_trk(tract_path, nii_file, Space.RASMM)
    streamlines = streams.get_streamlines_copy()

    return streamlines


def save_ft(folder_name, subj_name, streamlines, nii_file, file_name = "_wholebrain.trk"):
    from dipy.io.streamline import save_trk
    from dipy.io.stateful_tractogram import StatefulTractogram, Space

    dir_name = f'{folder_name}{os.sep}streamlines'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    tract_name = dir_name + subj_name + file_name
    save_trk(StatefulTractogram(streamlines,nii_file,Space.RASMM),tract_name)


def non_weighted_con_mat(streamlines, lab_labels_index, affine, folder_name, fig_type=''):
    from dipy.tracking import utils
    m, grouping = utils.connectivity_matrix(streamlines, affine, lab_labels_index,
                                            return_mapping=True,
                                            mapping_as_streamlines=True)
    #remove '0' label from mat:
    mm = m[1:]
    mm = mm[:,1:]

    np.save(f'{folder_name}{os.sep}{fig_type}' , mm)

    return mm, grouping






