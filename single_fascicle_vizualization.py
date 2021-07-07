
from dipy.viz import window, actor
import numpy as np
from weighted_tracts import *
from remove_cci_outliers import remove_cci_outliers


def show_cc_parts_weighted(streamlines_g,streamlines_b,streamlines_s, g_mean, b_mean, s_mean, folder_name, lut_cmap, bar):

    mean_g_vec = [g_mean]*streamlines_g.__len__()
    mean_b_vec = [b_mean]*streamlines_b.__len__()
    mean_s_vec = [s_mean]*streamlines_s.__len__()


    genu_actor = actor.line(streamlines_g, mean_g_vec, linewidth=0.5,lookup_colormap=lut_cmap)

    body_actor = actor.line(streamlines_b, mean_b_vec, linewidth=0.5,lookup_colormap=lut_cmap)

    splenium_actor = actor.line(streamlines_s, mean_s_vec, linewidth=0.5,lookup_colormap=lut_cmap)

    r = window.Renderer()
    r.add(genu_actor)
    r.add(body_actor)
    r.add(splenium_actor)
    r.add(bar)

    window.show(r)

    save_as = folder_name + '\cc_parts.png'
    r.set_camera(r.camera_info())
    window.record(r, out_path=save_as, size=(800, 800))


def calc_mean_cc_vals(fascicle_name):
    mean_vals_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\mean_vals'
    fascicle_path = mean_vals_folder+ fascicle_name+'_vals.npy'
    fascicle = np.load(fascicle_path)
    g,b,s = np.nanmean(fascicle,axis=0)

    return g,b,s


def cc_part_viz_running_script(n,folder_name):
    folder_name = folder_name +r'\streamlines'

    hue = [0.0,1.0]
    saturation = [0.0,1.0]
    scale = [3,7]

    lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                           saturation_range=saturation, scale_range=scale)
    bar = actor.scalar_bar(lut_cmap)

    fascicle_name = '\cc_parts'
    g_mean,b_mean,s_mean = calc_mean_cc_vals(fascicle_name)


    g_path = folder_name + n + r'_genu_cortex_cleaned.trk'
    streamlines_g = load_ft(g_path)
    b_path = folder_name + n + r'_body_cortex_cleaned.trk'
    streamlines_b = load_ft(b_path)
    s_path = folder_name + n + r'_splenium_cortex_cleaned.trk'
    streamlines_s = load_ft(s_path)


    show_cc_parts_weighted(streamlines_g,streamlines_b,streamlines_s, g_mean, b_mean, s_mean, folder_name, lut_cmap, bar)


def create_streamline_dict(streamlines, lab_labels_index, affine):
    from dipy.tracking import utils

    m, grouping = utils.connectivity_matrix(streamlines, affine, lab_labels_index,
                                            return_mapping=True,
                                            mapping_as_streamlines=True)

    return grouping


def load_mat_of_median_vals(mat_type = 'w_plus'):

    median_vals_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\mean_vals\plus3'
    mat_path = median_vals_folder+ '\mean_'+ mat_type+'.npy'
    #mat_path = r'C:\Users\Admin\my_scripts\Ax3D_Pack\mean_vals\plus\weighted_mega_wholebrain_plus_new2_nonnorm.npy'
    mat_medians = np.load(mat_path)

    return mat_medians


def nodes_by_index_megaaa(folder_name):
    import nibabel as nib
    lab = folder_name + r'\rMegaAtlas_Labels.nii'
    lab_file = nib.load(lab)
    lab_labels = lab_file.get_data()
    affine = lab_file.affine
    lab_labels_index = [labels for labels in lab_labels]
    lab_labels_index = np.asarray(lab_labels_index, dtype='int')
    return lab_labels_index, affine


def clean_non_cc(grouping):
    clean_grouping = {}
    for pair, tracts in grouping.items():
        if pair[0] == 0 or pair[1] == 0:
            continue
        else:
            if pair[0] in list(np.arange(1,51)) and pair[1] in list(np.arange(51,101)):
                clean_grouping[pair] = tracts
            if pair[1] in list(np.arange(1,51)) and pair[0] in list(np.arange(51,101)):
                clean_grouping[pair] = tracts
    return clean_grouping


def choose_specific_bundle(streamlines, affine, folder_name,mask_type, n, nii_file):
    from dipy.tracking import utils
    from dipy.tracking.streamline import Streamlines


    #masks_dict = {'cc':[1,2,3,4,5],'cr':[6],'cing':[7],'mcp':[8],'fx':[9]}
    file_list = os.listdir(folder_name)
    for file in file_list:
        if 'ratlas' in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            all_mask_mat = mask_img.get_data()

        if 'CR_mask' in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            cr_mask_mat = mask_img.get_fdata()

        if 'IFOF_mask' in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            ifof_mask_mat = mask_img.get_data()

        if 'ILF_mask' in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            ilf_mask_mat = mask_img.get_data()

        if 'slf_mask' in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            slf_mask_mat = mask_img.get_data()

        if 'cc1_mask' in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            cc_mask_mat = mask_img.get_fdata()

        if 'fx_mask' in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            fx_mask_mat = mask_img.get_data()

        if 'midsag_mask' in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            midsag_mask_mat = mask_img.get_data()

        if 'no_left_mask' in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            left_out_mask_mat = mask_img.get_data()

        if 'no_right_mask' in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            right_out_mask_mat = mask_img.get_data()

    if mask_type=='w':
        masked_streamlines = streamlines
        return masked_streamlines

    elif mask_type=='cc':
        mask_include = cc_mask_mat == 1
        mask_exclude =(ifof_mask_mat == 1)|(cc_mask_mat == 2)|(ilf_mask_mat == 1)|(slf_mask_mat == 1)|(cr_mask_mat==1)|(fx_mask_mat==1)

        #mask_include = (all_mask_mat > 0) & (all_mask_mat < 6)
        #mask_exclude = (all_mask_mat > 5)

    elif mask_type=='cr':
        mask_include = (cr_mask_mat == 1)|(cr_mask_mat ==3)
        mask_exclude = (cr_mask_mat == 2)| (right_out_mask_mat ==1)

    elif mask_type=='cing':
        mask_include = (all_mask_mat == 7)
        mask_exclude = (midsag_mask_mat == 1) | (left_out_mask_mat ==1)

    elif mask_type=='mcp':
        mask_include = (all_mask_mat == 8)
        mask_exclude = (all_mask_mat < 6)

    elif mask_type=='fx':
        mask_include = (fx_mask_mat == 1)#(left_out_mask_mat ==1)|(right_out_mask_mat ==1)
        mask_exclude = (cr_mask_mat == 1) | (ifof_mask_mat==1) | (ilf_mask_mat==1)|(all_mask_mat == 7)|(slf_mask_mat == 1)#(cc_mask_mat == 1)|

    elif mask_type == 'ifof':
        mask_include = (ifof_mask_mat == 1)
        mask_exclude = (midsag_mask_mat == 1) | (cr_mask_mat == 1) | (ilf_mask_mat == 1)| (slf_mask_mat == 1)| (ifof_mask_mat==2) #| (left_out_mask_mat ==1)

    elif mask_type == 'ilf':
        mask_include = (ilf_mask_mat == 1)
        mask_exclude = (midsag_mask_mat == 1) | (cr_mask_mat == 1) | (ifof_mask_mat==1) | (ilf_mask_mat==2) | (right_out_mask_mat ==1)

    elif mask_type == 'slf':
        mask_include = (slf_mask_mat == 1)
        mask_exclude = (midsag_mask_mat == 1) | (cr_mask_mat == 1)|(slf_mask_mat == 2) | (right_out_mask_mat ==1)


    masked_streamlines = utils.target(streamlines, affine, mask_include)
    masked_streamlines = utils.target(masked_streamlines, affine, mask_exclude, include=False)

    masked_streamlines = Streamlines(masked_streamlines)
    save_ft(folder_name, n, masked_streamlines, nii_file, file_name='_'+mask_type+'.trk')
    return masked_streamlines


def find_bundle(folder_name, n, nii_file, mask_type):
    tract_path = folder_name+r'\streamlines'+n+'_wholebrain_3d_plus_new.trk'
    streamlines = load_ft(tract_path, nii_file)
    affine = nodes_by_index_mega(folder_name)[1]
    masked_streamlines = choose_specific_bundle(streamlines, affine, folder_name, mask_type, n, nii_file)

    return masked_streamlines


def streamline_mean_fascicle_value_weighted(folder_name, n, nii_file, mask_type,masked_streamlines, weight_by='1.5_2_AxPasi5'):

    lab_labels_index, affine = nodes_by_index_general(folder_name)
    streamline_dict = create_streamline_dict(masked_streamlines, lab_labels_index, affine)
    bvec_file = load_dwi_files(folder_name)[6]
    idx = nodes_labels_mega(index_to_text_file)[1]

    mat_medians = weighted_con_mat_mega(bvec_file, weight_by, streamline_dict, idx, folder_name, fig_type='')[1]
    id = np.argsort(idx)
    mat_medians = mat_medians[id]
    mat_medians = mat_medians[:, id]
    vec_vols = []
    s_list=[]
    '''new func:'''
    for i in range(id.__len__()): #
        for j in range(i+1):  #
            edge_s_list = []
            #print(i,j)
            if (i + 1, j + 1) in streamline_dict and mat_medians[i, j] > 0:
                edge_s_list += streamline_dict[(i + 1, j + 1)]
            if (j + 1, i + 1) in streamline_dict and mat_medians[i, j] > 0:
                edge_s_list += streamline_dict[(j + 1, i + 1)]
            edge_vec_vols = [mat_medians[i,j]]*edge_s_list.__len__()

            s_list = s_list+edge_s_list
            vec_vols = vec_vols+edge_vec_vols

    return s_list, vec_vols



def show_fascicles_wholebrain(s_list, vec_vols, folder_name, mask_type, downsamp=1, scale=[3, 6], hue=[0.25, -0.05],saturation = [0.1, 1.0]):

    s_img = folder_name+r'\streamlines'+r'\fascicles_AxCaliber_weighted_3d_'+mask_type+'.png'
    #hue = [0.4, 0.7] # blues
    #hue = [0.25, -0.05] #Hot
    #hue = [0, 1] #All

    weighted=True
    '''
    if weighted:
        scale = [0, 3]


    else:
        scale = [0, 6]
        vec_vols = np.log(vec_vols)
        #vec_vols = vec_vols-np.nanmin(vec_vols)/(np.nanmax(vec_vols)-np.nanmin(vec_vols))
    '''

    if downsamp != 1:
        vec_vols = vec_vols[::downsamp]
        s_list = s_list[::downsamp]

    lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                           saturation_range=saturation, scale_range=scale)
    bar = actor.scalar_bar(lut_cmap)
    #w_actor = actor.line(s_list, vec_vols, linewidth=1.2, lookup_colormap=lut_cmap)
    w_actor = actor.streamtube(s_list, vec_vols, linewidth=0.6, lookup_colormap=lut_cmap)

    #w_actor = actor.streamtube(s_list, vec_vols, linewidth=0.3, lookup_colormap=lut_cmap)
    #w_actor = actor.line(s_list, linewidth=1.0, lookup_colormap=lut_cmap)


    r = window.Renderer()
    #r.SetBackground(*window.colors.white)
    r.add(w_actor)
    r.add(bar)
    window.show(r)
    r.set_camera(r.camera_info())
    window.record(r, out_path=s_img, size=(800, 800))


if __name__ == '__main__':

    main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep'
    folder_name = main_folder+all_subj_folders[8]
    n = all_subj_names[8]
    nii_file = load_dwi_files(folder_name)[5]

    #s_list, vec_vols = streamline_mean_fascicle_value_weighted(folder_name, n, nii_file,'cc')
    #cc_part_viz_running_script(n,folder_name)
    tract_path = folder_name+r'\streamlines'+n+'_wholebrain_3d_plus_new.trk'

    streamlines = load_ft(tract_path, nii_file)

    masked_streamlines = find_bundle(folder_name, n, nii_file, 'slfn')
    lab_labels_index, affine = nodes_by_index_general(folder_name)
    masked_streamlines = choose_specific_bundle(streamlines, affine, folder_name, 'slf', n, nii_file, masked_streamlines)



