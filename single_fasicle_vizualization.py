
from dipy.viz import window, actor
from FT.all_subj import all_subj_folders, all_subj_names
import numpy as np
from FT.weighted_tracts import *

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

    m, grouping = utils.connectivity_matrix(streamlines, lab_labels_index, affine=affine,
                                            return_mapping=True,
                                            mapping_as_streamlines=True)

    return grouping

def load_mat_of_median_vals(mat_type = 'w_AxCaliber'):

    median_vals_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\mean_vals'
    mat_path = median_vals_folder+ '\mean_'+ mat_type+'_nonnorm.npy'
    mat_medians = np.load(mat_path)

    return mat_medians


def streamline_mean_fascicle_value_weighted(folder_name, n):
    from dipy.tracking.streamline import Streamlines

    tract_path = folder_name+r'\streamlines'+n+'_wholebrain_7d.trk'
    streamlines = load_ft(tract_path)
    lab_labels_index, affine = nodes_by_index_mega(folder_name)
    streamline_dict = create_streamline_dict(streamlines, lab_labels_index, affine)
    mat_medians =  load_mat_of_median_vals(mat_type = 'w_AxCaliber')

    #new func:
    vec_vols = []
    s_list=[]
    for i in range(mat_medians.shape[0]):
        for j in range(i+1):
            if np.nonzero(mat_medians[i, j]):
                edge_s_list = streamline_dict[(i+1,j+1)]
                edge_vec_vols = [mat_medians[i,j]]*edge_s_list.__len__()
                vec_vols = vec_vols+edge_vec_vols
                s_list = s_list+edge_s_list


    s_list = Streamlines(s_list)
    show_fascicles_wholebrain(s_list, vec_vols)


def show_fascicles_wholebrain(s_list, vec_vols):
    hue = [0.0, 1.0]
    saturation = [0.0, 1.0]
    scale = [0, 5.5]
    lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                           saturation_range=saturation, scale_range=scale)
    bar = actor.scalar_bar(lut_cmap)
    w_actor = actor.line(s_list, vec_vols, linewidth=0.5, lookup_colormap=lut_cmap)
    r = window.Renderer()
    r.add(w_actor)
    r.add(bar)
    window.show(r)


if __name__ == '__main__':

    main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep'
    folder_name = main_folder+all_subj_folders[0]
    n = all_subj_names[0]

    streamline_mean_fascicle_value_weighted(folder_name, n)
    #cc_part_viz_running_script(n,folder_name)


