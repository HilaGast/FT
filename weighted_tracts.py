import os
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
            labels_file_name = os.path.join(folder_name, file)
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
        if 'mask' in file and mask_type in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            mask_mat = mask_img.get_data()
    return mask_mat


def create_seeds(folder_name, white_matter, affine, use_mask = True, mask_type='cc'):
    if use_mask:
        mask_mat = load_mask(folder_name,mask_type)
        seed_mask = mask_mat == 1
    else:
        seed_mask = white_matter
    seeds = utils.seeds_from_mask(seed_mask, density=2, affine=affine)
    return seeds


def create_csd_model(data, gtab, white_matter, sh_order=6):
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

    csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=sh_order)
    csd_fit = csd_model.fit(data, mask=white_matter)

    return csd_fit


def create_fa_classifier(gtab,data,white_matter):
    import dipy.reconst.dti as dti
    from dipy.reconst.dti import fractional_anisotropy

    tensor_model = dti.TensorModel(gtab)
    tenfit = tensor_model.fit(data, mask=white_matter)
    fa = fractional_anisotropy(tenfit.evals)
    classifier = ThresholdTissueClassifier(fa, .22)

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


def weighting_streamlines(streamlines, nii_file, weight_by = 'pasiS',hue = [0.0,1.0],saturation = [0.0,1.0], scale = [0,6]):
    '''
    weight_by = 'pasiS'
    hue = [0.0,1.0]
    saturation = [0.0,1.0]
    scale = [0,6]
    '''
    from dipy.viz import window, actor, colormap as cmap
    from dipy.tracking.streamline import transform_streamlines,values_from_volume

    weight_by_file = nii_file[:-4:]+'_'+weight_by+'.nii'
    weight_by_img = nib.load(weight_by_file)
    weight_by_data = weight_by_img.get_data()
    affine = weight_by_img.get_affine()
    stream = list(streamlines)
    vol_per_tract = values_from_volume(weight_by_data, stream, affine=affine)
    pfr_file = nii_file[:-4:] + '_pfrS.nii'
    pfr_img = nib.load(pfr_file)
    pfr_data = pfr_img.get_data()
    pfr_per_tract = values_from_volume(pfr_data, stream, affine=affine)

    #Leave out from the calculation of mean value per tract, a chosen quantile:
    vol_vec = weight_by_data.flatten()
    q = np.quantile(vol_vec[vol_vec>0], 0.95)
    mean_vol_per_tract = []
    for s, pfr in zip(vol_per_tract, pfr_per_tract):
        s = np.asanyarray(s)
        non_out = [s < q]
        pfr = np.asanyarray(pfr)
        high_pfr = [pfr > 60]
        mean_vol_per_tract.append(np.mean(s[tuple(non_out and high_pfr)]))

    lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                           saturation_range=saturation, scale_range=scale)
    streamlines_actor = actor.line(streamlines, mean_vol_per_tract, linewidth=0.6, lookup_colormap=lut_cmap)
    bar = actor.scalar_bar(lut_cmap)
    r = window.Renderer()
    r.add(streamlines_actor)
    r.add(bar)
    mean_pasi_weighted_img = folder_name+'\streamlines\mean_pasi_weighted.png'
    window.show(r)
    r.set_camera(r.camera_info())
    window.record(r, out_path=mean_pasi_weighted_img, size=(800, 800))


def load_ft(tract_path):
    from dipy.io.streamline import load_trk
    from dipy.tracking.streamline import Streamlines

    streams, hdr = load_trk(tract_path)
    streamlines = Streamlines(streams)

    return streamlines


def save_ft(folder_name,streamlines,affine, labels):
    from dipy.io.streamline import save_trk

    dir_name = folder_name + '\streamlines'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    tract_name = os.path.join(dir_name, (folder_name.split(sep="\\")[-1] + ".trk")) #I should change it to more general script, the name is the last path part.
    save_trk(tract_name, streamlines, affine=np.eye(4), shape=labels.shape, vox_size=np.array([1.7,1.7,1.7]))


def weighted_connectivity_matrix(streamlines, folder_name, nii_file, weight_by='pasiS'):
    import matplotlib.pyplot as plt
    from dipy.tracking import utils
    from dipy.tracking.streamline import values_from_volume

    lab = folder_name+r'\rAAL_highres_atlas.nii'
    lab_file = nib.load(lab)
    lab_labels = lab_file.get_data()
    affine = lab_file.get_affine()
    uni = np.unique(lab_labels)
    lab_labels_index = lab_labels
    for index, i in enumerate(uni):
        lab_labels_index[lab_labels_index == i] = index
    labels_file = open(r'C:\Users\Admin\my_scripts\aal\origin\aal2nii.txt', 'r', errors='ignore')
    labels_name = labels_file.readlines()
    labels_table = []
    labels_headers = []
    for line in labels_name:
        if not line[0] == '#':
            labels_table.append([col for col in line.split(" ") if col])
        elif 'ColHeaders' in line:
            labels_headers = [col for col in line.split(" ") if col]
            labels_headers = labels_headers[2:]
    labels_file.close()
    for i, l in enumerate(labels_table):
        labels_headers.append(l[1])

    # non-weighted:
    m, grouping = utils.connectivity_matrix(streamlines, lab_labels_index, affine=affine,
                                            return_mapping=True,
                                            mapping_as_streamlines=True)
    log_m = np.log1p(m)
    new_data = np.zeros(np.array(log_m.shape) * 5)
    for j in range(log_m.shape[0]):
        for k in range(log_m.shape[1]):
            new_data[j * 5: (j + 1) * 5, k * 5: (k + 1) * 5] = log_m[j, k]

    plt.imshow(new_data, interpolation='nearest', cmap='hot', origin='upper')
    plt.colorbar()
    #plt.xlabel(labels_headers)
    plt.title('Non-weighted connectivity matrix', fontsize=16)
    plt.savefig(folder_name+r'\non-weighted(whole brain).png')
    plt.show()

    # weighted:
    weight_by_file = nii_file[:-4:]+'_'+weight_by+'.nii'
    weight_by_img = nib.load(weight_by_file)
    weight_by_data = weight_by_img.get_data()
    affine = weight_by_img.get_affine()
    m_weighted = np.zeros((117, 117), dtype='int64')
    for pair,tracts in grouping.items():
        mean_vol_per_tract = []
        vol_per_tract = values_from_volume(weight_by_data, tracts, affine=affine)
        for s in vol_per_tract:
            mean_vol_per_tract.append(np.mean(s))
        mean_path_vol = np.mean(mean_vol_per_tract)
        m_weighted[pair[0], pair[1]] = mean_path_vol
        m_weighted[pair[1], pair[0]] = mean_path_vol

    log_m = np.log1p(m_weighted)
    new_data = np.zeros(np.array(log_m.shape) * 5)
    for j in range(log_m.shape[0]):
        for k in range(log_m.shape[1]):
            new_data[j * 5: (j + 1) * 5, k * 5: (k + 1) * 5] = log_m[j, k]

    plt.imshow(new_data, interpolation='nearest', cmap='hot', origin='upper')
    plt.colorbar()
    #plt.xlabel(labels_headers)
    plt.title('Weighted connectivity matrix', fontsize=16)
    plt.savefig(folder_name+r'\weighted(whole brain).png')
    plt.show()


if __name__ == '__main__':
    folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5\NaYa_subj9'
    mask_type = 'cc'
    gtab, data, affine, labels, white_matter, nii_file = load_dwi_files(folder_name)
    mask_mat = load_mask(folder_name,mask_type)
    seeds = create_seeds(folder_name, white_matter, affine, use_mask=False, mask_type='cc')
    csd_fit = create_csd_model(data, gtab, white_matter, sh_order=6)
    fa, classifier = create_fa_classifier(gtab, data, white_matter)
    streamlines = create_streamlines(csd_fit, classifier, seeds, affine)
    #weighting_streamlines(streamlines, nii_file, weight_by='pasiS', hue=[0.0, 1.0], saturation=[0.0, 1.0], scale=[0, 10])
    #save_ft(folder_name,streamlines,affine, labels)
    weighted_connectivity_matrix(streamlines, folder_name, nii_file, weight_by='pasiS')
