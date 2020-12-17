import os
import nibabel as nib
from dipy.tracking import utils
from dipy.core.gradients import gradient_table
from dipy.tracking.local_tracking import LocalTracking
import numpy as np
from all_subj import *


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
    nii_file = os.path.join(folder_name,'diff_corrected.nii')
    hardi_img = nib.load(nii_file)
    data = hardi_img.get_fdata()
    affine = hardi_img.affine
    gtab = gradient_table(bval_file, bvec_file, small_delta=small_delta)
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
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel,auto_response
    response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

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


def create_act_classifier(fa,folder_name,labels):  # Does not working
    from dipy.tracking.stopping_criterion import ActStoppingCriterion
    background = np.ones(labels.shape)
    background[(np.asarray(labels)>0) > 0] = 0
    include_map = np.zeros(fa.shape)
    lab = f'{folder_name}{os.sep}rMegaAtlas_cortex_Labels.nii'
    lab_file = nib.load(lab)
    lab_labels = lab_file.get_data()
    include_map[background>0] = 1
    include_map[lab_labels > 0] = 1
    include_map[fa>0.18] = 1
    include_map = include_map==1
    exclude_map = labels==1

    act_classifier = ActStoppingCriterion(include_map, exclude_map)

    return act_classifier


def create_streamlines(csd_fit, classifier, seeds, affine):
    from dipy.data import default_sphere
    from dipy.direction import DeterministicMaximumDirectionGetter
    from dipy.tracking.streamline import Streamlines
    print('Starting to compute streamlines')
    detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                                 max_angle=30.,
                                                                 sphere=default_sphere)

    streamlines = Streamlines(LocalTracking(detmax_dg, classifier, seeds, affine, step_size=1,return_all=False))

    long_streamlines = np.ones((len(streamlines)), bool)
    for i in range(0, len(streamlines)):
        if streamlines[i].shape[0] < 100:
            long_streamlines[i] = False
    streamlines = streamlines[long_streamlines]

    return streamlines


def weighting_streamlines(folder_name, streamlines, bvec_file, show=False, weight_by = '1.5_2_AxPasi5',hue = [0.0,1.0],saturation = [0.0,1.0], scale = [2,7],fig_type=''):
    '''
    weight_by = '1.5_2_AxPasi5'
    hue = [0.0,1.0]
    saturation = [0.0,1.0]
    scale = [3,6]
    '''
    from dipy.tracking.streamline import values_from_volume

    weight_by_data, affine = load_weight_by_img(folder_name,weight_by)

    stream = list(streamlines)
    vol_per_tract = values_from_volume(weight_by_data, stream, affine=affine)

    pfr_data = load_weight_by_img(folder_name,'1.5_2_AxFr5')[0]

    pfr_per_tract = values_from_volume(pfr_data, stream, affine=affine)

    #Leave out from the calculation of mean value per tract, a chosen quantile:
    vol_vec = weight_by_data.flatten()
    q = np.quantile(vol_vec[vol_vec>0], 0.95)
    mean_vol_per_tract = []
    for s, pfr in zip(vol_per_tract, pfr_per_tract):
        s = np.asanyarray(s)
        non_out = [s < q]
        pfr = np.asanyarray(pfr)
        high_pfr = [pfr > 0.5]
        mean_vol_per_tract.append(np.nanmean(s[tuple(non_out and high_pfr)]))

    if show:
        show_tracts(hue,saturation,scale,streamlines,mean_vol_per_tract,folder_name,fig_type)

    return mean_vol_per_tract


def show_tracts(hue,saturation,scale,streamlines,mean_vol_per_tract,folder_name,fig_type):
    from dipy.viz import window, actor
    lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                           saturation_range=saturation, scale_range=scale)
    streamlines_actor = actor.line(streamlines, mean_vol_per_tract, linewidth=1, lookup_colormap=lut_cmap)
    bar = actor.scalar_bar(lut_cmap)
    r = window.Scene()
    r.add(streamlines_actor)
    r.add(bar)
    mean_pasi_weighted_img = f'{folder_name}{os.sep}streamlines{os.sep}mean_pasi_weighted{fig_type}.png'
    window.show(r)
    r.set_camera(r.camera_info())
    window.record(r, out_path=mean_pasi_weighted_img, size=(800, 800))


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


def nodes_by_index(folder_name):
    import numpy as np
    import nibabel as nib
    lab = f'{folder_name}{os.sep}rMegaAtlas_Labels_highres.nii'
    lab_file = nib.load(lab)
    lab_labels = lab_file.get_fdata()
    affine = lab_file.affine
    uni = np.unique(lab_labels)
    lab_labels_index = lab_labels
    for index, i in enumerate(uni):
        lab_labels_index[lab_labels_index == i] = index
    return lab_labels_index, affine


def nodes_by_index_general(folder_name,atlas='mega'):
    import nibabel as nib
    if atlas == 'mega':
        lab = f'{folder_name}{os.sep}rMegaAtlas_Labels_highres.nii'
    elif atlas == 'aal3':
        lab = folder_name + r'\rAAL3_highres_atlas.nii'
        #lab = folder_name + r'\rAAL3_highres_atlas_corrected.nii'
    elif atlas == 'yeo7_200':
        lab = folder_name + r'\ryeo7_200_atlas.nii'

    lab_file = nib.load(lab)
    lab_labels = lab_file.get_fdata()
    affine = lab_file.affine
    lab_labels_index = [labels for labels in lab_labels]
    lab_labels_index = np.asarray(lab_labels_index, dtype='int')
    return lab_labels_index, affine


def nodes_labels_aal3(index_to_text_file):
    labels_file = open(index_to_text_file, 'r', errors='ignore')
    labels_name = labels_file.readlines()
    labels_file.close()
    labels_table = []
    labels_headers = []
    idx = []
    for line in labels_name:
        if not line[0] == '#':
            labels_table.append([col for col in line.split() if col])

    for l in labels_table:
        if len(l)==3:
            head = l[1]
            labels_headers.append(head)
            idx.append(int(l[0])-1)
    #pop over not assigned indices (in aal3):
    idx = np.asarray(idx)
    first=idx>35
    second=idx>81
    idx[first]-=2
    idx[second]-=2
    idx=list(idx)
    #removeidx = [82,81,36,35]
    #for i in removeidx:
    #    del labels_headers[i]

    return labels_headers, idx


def nodes_labels_yeo7(index_to_text_file):
    labels_file = open(index_to_text_file, 'r', errors='ignore')
    labels_name = labels_file.readlines()
    labels_file.close()
    labels_table = []
    labels_headers = []
    idx = []
    for line in labels_name:
        if not line[0] == '#':
            labels_table.append([col for col in line.split() if col])

    for l in labels_table:
        if len(l)>= 3:
            head = l[1]
            labels_headers.append(head)
            idx.append(int(l[0])-1)

    idx=list(idx)

    return labels_headers, idx


def nodes_labels_mega(index_to_text_file):
    labels_file = open(index_to_text_file, 'r', errors='ignore')
    labels_name = labels_file.readlines()
    labels_file.close()
    labels_table = []
    labels_headers = []
    idx = []
    for line in labels_name:
        if not line[0] == '#':
            labels_table.append([col for col in line.split("\t") if col])
        elif 'ColHeaders' in line:
            labels_headers = [col for col in line.split(" ") if col]
            labels_headers = labels_headers[2:]
    for l in labels_table:
        head = l[1]
        labels_headers.append(head[:-1])
        idx.append(int(l[0])-1)
    return labels_headers, idx


def non_weighted_con_mat_mega(streamlines, lab_labels_index, affine, idx, folder_name, fig_type=''):
    from dipy.tracking import utils

    if len(fig_type) >> 0:
        fig_type = '_'+fig_type

    m, grouping = utils.connectivity_matrix(streamlines, affine, lab_labels_index,
                                            return_mapping=True,
                                            mapping_as_streamlines=True)
    mm = m[1:]
    mm = mm[:,1:]
    if 'aal3' in fig_type:
        mm = np.delete(mm, [34, 35, 80, 81], 0)
        mm = np.delete(mm, [34, 35, 80, 81], 1)

    mm = mm[idx]
    mm = mm[:, idx]
    new_data = 1 / mm  # values distribute between 0 and 1, 1 represents distant nodes (only 1 tract)
    #new_data[new_data > 1] = 2
    #np.save(folder_name + r'\non-weighted_mega'+fig_type, new_data)
    np.save(folder_name + r'\non-weighted'+fig_type+'_nonnorm', mm)

    return new_data, m, grouping


def weighted_con_mat_mega(bvec_file, weight_by, grouping, idx, folder_name,fig_type=''):
    from dipy.tracking.streamline import values_from_volume
    import numpy as np

    if len(fig_type) >> 0:
        fig_type = '_' + fig_type


    weight_by_data, affine = load_weight_by_img(folder_name,weight_by)

    pfr_data = load_weight_by_img(folder_name,'1.5_2_AxFr5')[0]

    vol_vec = weight_by_data.flatten()
    q = np.quantile(vol_vec[vol_vec>0], 0.95)
    m_weighted = np.zeros((len(idx),len(idx)), dtype='float64')
    for pair, tracts in grouping.items():
        if pair[0] == 0 or pair[1] == 0:
            continue
        else:
            mean_vol_per_tract = []
            vol_per_tract = values_from_volume(weight_by_data, tracts, affine=affine)
            pfr_per_tract = values_from_volume(pfr_data, tracts, affine=affine)
            for s, pfr in zip(vol_per_tract, pfr_per_tract):
                s = np.asanyarray(s)
                non_out = [s < q]
                pfr = np.asanyarray(pfr)
                high_pfr = [pfr > 0.5]
                mean_vol_per_tract.append(np.nanmean(s[tuple(non_out and high_pfr)]))
            mean_path_vol = np.nanmean(mean_vol_per_tract)
            if 'aal3' in fig_type:
                r= pair[0]-1
                c=pair[1]-1

                if r>81:
                    r-=4
                elif r>35:
                    r-=2

                if c>81:
                    c-=4
                elif c>35:
                    c-=2

                m_weighted[r,c] = mean_path_vol
                m_weighted[c,r] = mean_path_vol

            else:
                m_weighted[pair[0]-1, pair[1]-1] = mean_path_vol
                m_weighted[pair[1]-1, pair[0]-1] = mean_path_vol

    mm_weighted = m_weighted[idx]
    mm_weighted = mm_weighted[:, idx]
    #mm_weighted[mm_weighted<0.01] = 0
    new_data = 1/(mm_weighted*8.75) #8.75 - axon diameter 2 ACV constant
    #new_data[new_data ==1] = 2
    #if "AxPasi" in weight_by:
    #np.save(folder_name + r'\weighted_mega'+fig_type, new_data)
    np.save(folder_name + r'\weighted'+fig_type+'_nonnorm', mm_weighted)


    return new_data, mm_weighted


def draw_con_mat(data, h, fig_name, is_weighted=False):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    max_val = np.max(data[np.isfinite(data)])
    data[~np.isfinite(data)] = np.nan
    if is_weighted:
        #data[data>0.8*max_val] = 0.8*max_val
        data[~np.isfinite(data)] = max_val
        mat_title = 'AxCaliber weighted connectivity matrix'
        plt.figure(1, [30, 24])
        cmap = cm.YlOrRd
        cmap.set_over('black')
        plt.imshow(data, interpolation='nearest', cmap=cmap, origin='upper',vmax=0.99*max_val)
        plt.colorbar()
        plt.xticks(ticks=np.arange(0, len(data), 1), labels=h)
        plt.yticks(ticks=np.arange(0, len(data), 1), labels=h)
        plt.title(mat_title, fontsize=32)
        plt.tick_params(axis='x', pad=10.0, labelrotation=90, labelsize=11)
        plt.tick_params(axis='y', pad=10.0, labelsize=11)
        plt.savefig(fig_name)
        plt.show()
    else:
        data[~np.isfinite(data)] = max_val
        mat_title = 'Number of tracts weighted connectivity matrix'
        plt.figure(1, [30, 24])
        cmap = cm.YlOrRd
        cmap.set_over('black')
        plt.imshow(data, interpolation='nearest', cmap=cmap, origin='upper',vmax=0.99*max_val, norm = colors.LogNorm(vmax=0.99*max_val))
        plt.colorbar()
        plt.xticks(ticks=np.arange(0, len(data), 1), labels=h)
        plt.yticks(ticks=np.arange(0, len(data), 1), labels=h)
        plt.title(mat_title, fontsize=32)
        plt.tick_params(axis='x', pad=10.0, labelrotation=90, labelsize=11)
        plt.tick_params(axis='y', pad=10.0, labelsize=11)
        plt.savefig(fig_name)
        plt.show()


def weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type = 'whole brain', weight_by='1.5_2_AxPasi5'):

    lab_labels_index, affine = nodes_by_index_general(folder_name,atlas='yeo7_200')
    labels_headers, idx = nodes_labels_yeo7(index_to_text_file)

    # non-weighted:

    new_data, m, grouping = non_weighted_con_mat_mega(streamlines, lab_labels_index, affine, idx, folder_name, fig_type)
    h = labels_headers
    #fig_name = folder_name + r'\non-weighted('+fig_type+', MegaAtlas).png'
    #fig_name = folder_name + r'\non-weighted(' + fig_type + ', AAL3).png'
    fig_name = folder_name + r'\non-weighted(' + fig_type + ', yeo7,200).png'
    #draw_con_mat(new_data, h, fig_name, is_weighted=False)

    # weighted:

    new_data, mm_weighted = weighted_con_mat_mega(bvec_file, weight_by, grouping, idx, folder_name, fig_type)
    #fig_name = folder_name + r'\Weighted('+fig_type+', MegaAtlas).png'
    #fig_name = folder_name + r'\Weighted('+fig_type+', AAL3).png'
    fig_name = folder_name + r'\Weighted(' + fig_type + ', yeo7,200).png'
    draw_con_mat(new_data, h, fig_name, is_weighted=True)


def load_weight_by_img(folder_name, weight_by):
    import nibabel as nib
    for file in os.listdir(folder_name):
        if weight_by in file and file.endswith('.nii'):
            weight_by_file = os.path.join(folder_name,file)
            continue
    weight_by_img = nib.load(weight_by_file)
    weight_by_data = weight_by_img.get_fdata()
    affine = weight_by_img.affine

    return weight_by_data, affine


def streamlins_len_connectivity_mat(folder_name, streamlines, lab_labels_index, idx, fig_type='lengths'):
    m, grouping = utils.connectivity_matrix(streamlines, affine, lab_labels_index, return_mapping = True, mapping_as_streamlines = True)
    new_m = np.zeros(m.shape)
    new_grouping = grouping.copy()
    for k, v in new_grouping.items():
        if k[0]==0 or k[1]==0:
            continue
        lengths = []
        for stream in v:
            lengths.append(stream.shape[0])
        new_m[k[0] - 1, k[1] - 1] = np.mean(lengths)
        new_m[k[1] - 1, k[0] - 1] = np.mean(lengths)

    new_mm = new_m[idx]
    new_mm = new_mm[:, idx]
    np.save(folder_name + r'\weighted_mega_' + fig_type + '_nonnorm', new_mm)


def streamlines2groups_by_size(folder_name, n, streamlines, bvec_file, nii_file, first_cut=5.2, second_cut=6):
    import matplotlib.pyplot as plt
    mean_vol_per_tract = weighting_streamlines(folder_name, streamlines, bvec_file, show=False,
                                               weight_by='1.5_2_AxPasi5')
    mean_vol_per_tract = np.asarray(mean_vol_per_tract)
    sml_tracts_idx = mean_vol_per_tract <= first_cut
    med_tracts_idx = [first_cut < mean_vol_per_tract] and [mean_vol_per_tract < second_cut]
    lrg_tracts_idx = second_cut <= mean_vol_per_tract
    save_ft(folder_name, n, streamlines[sml_tracts_idx], nii_file, file_name="_sml_4d_labmask.trk")
    save_ft(folder_name, n, streamlines[med_tracts_idx], nii_file, file_name="_med_4d_labmask.trk")
    save_ft(folder_name, n, streamlines[lrg_tracts_idx], nii_file, file_name="_lrg_4d_labmask.trk")


if __name__ == '__main__':
    subj = all_subj_folders
    names = all_subj_names

    for s,n in zip(subj[::],names[::]):
        folder_name = subj_folder + s
        dir_name = folder_name + '\streamlines'
        gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
        #csd_fit = create_csd_model(data, gtab, white_matter, sh_order=6)
        #fa, classifier = create_fa_classifier(gtab, data, white_matter)
        lab_labels_index = nodes_by_index_general(folder_name,atlas='yeo7_200')[0]
        #seeds = create_seeds(folder_name, lab_labels_index, affine, use_mask=False, mask_type='cc', den=4)
        #streamlines = create_streamlines(csd_fit, classifier, seeds, affine)
        #save_ft(folder_name, n, streamlines, nii_file, file_name="_wholebrain_4d_labmask.trk")
        tract_path = f'{dir_name}{n}_wholebrain_4d_labmask.trk'
        idx = nodes_labels_yeo7(index_to_text_file)[1]
        streamlines = load_ft(tract_path, nii_file)
        weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='wholebrain_4d_labmask_yeo7_200_FA',
                                          weight_by='_FA')
        weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='wholebrain_4d_labmask_yeo7_200',
                                          weight_by='_AxPasi')
        #streamlins_len_connectivity_mat(folder_name, streamlines, lab_labels_index, idx, fig_type='lengths')
        #streamlines2groups_by_size(folder_name, n, streamlines, bvec_file, nii_file, first_cut=5.2, second_cut=6)



