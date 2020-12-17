
from weighted_tracts import *
from dipy.tracking.streamline import Streamlines
from fury.colormap import create_colormap
import matplotlib.cm as cm
from dipy.viz import window, actor
from advanced_interactive_viz import *


def create_streamline_dict(streamlines, lab_labels_index, affine):
    from dipy.tracking import utils

    m, grouping = utils.connectivity_matrix(streamlines, affine, lab_labels_index,
                                            return_mapping=True,
                                            mapping_as_streamlines=True)

    return grouping


def show_weighted_tractography(folder_name,vec_vols,s_list,bundle_short,direction,downsamp=1):
    s_img = rf'{folder_name}\streamlines\ax_fa_corr_{bundle_short}_{direction}.png'

    if downsamp != 1:
        vec_vols = vec_vols[::downsamp]
        s_list = s_list[::downsamp]
    vec_vols.append(1)
    vec_vols.append(-1)
    cmap = create_colormap(np.asarray(vec_vols),name='seismic')
    vec_vols=vec_vols[:-2]
    cmap=cmap[:-2]
    print(min(vec_vols),max(vec_vols))
    #w_actor = actor.line(s_list, vec_vols, linewidth=1.2, lookup_colormap=cmap)
    w_actor = actor.line(s_list,cmap, linewidth=1.2)
    r = window.Scene()
    #r.SetBackground(*window.colors.white)
    r.add(w_actor)
    #r.add(bar)
    window.show(r)
    r.set_camera(r.camera_info())
    window.record(r, out_path=s_img, size=(800, 800))


def load_vars(subj_folder,all_subj_folders, all_subj_names,i,bundle_name):
    main_folder = subj_folder
    folder_name = main_folder+all_subj_folders[i]
    n = all_subj_names[i]
    nii_file = load_dwi_files(folder_name)[5]
    #tract_path = folder_name+r'\streamlines'+n+'_wholebrain_4d_labmask.trk'
    tract_path = rf'{folder_name}\streamlines{n}_{bundle_name}.trk'

    streamlines = load_ft(tract_path, nii_file)
    lab_labels_index, affine = nodes_by_index_general(folder_name)

    #mat_weights = np.load(rf'{folder_name}\weighted_mega_wholebrain_4d_labmask_FA_nonnorm.npy')
    #mat_weights = np.load(rf'{folder_name}\AxCaliber_FA_r_no_th_50subj.npy')
    #mat_weights = np.load(rf'{folder_name}\weighted_mega_wholebrain_4d_labmask_nonnorm.npy')
    mat_weights = np.load(rf'{folder_name}\AxCaliber_FA_r_0.05_th_50subj.npy')

    idx = nodes_labels_mega(index_to_text_file)[1]
    id = np.argsort(idx)
    mat_weights = mat_weights[id]
    mat_weights = mat_weights[:,id]

    return streamlines, lab_labels_index, affine, id, mat_weights, folder_name


def weight_tracts_from_mat(id, streamline_dict, mat_weights):
    vec_vols = []
    s_list = []
    for col in range(id.__len__()):
        for row in range(col + 1):
            edge_s_list = []
            if (col + 1, row + 1) in streamline_dict:
                edge_s_list += streamline_dict[(col + 1, row + 1)]
            if (row + 1, col + 1) in streamline_dict:
                edge_s_list += streamline_dict[(row + 1, col + 1)]

            edge_vec_vols = [mat_weights[col, row]] * edge_s_list.__len__()
            s_list = s_list + edge_s_list
            vec_vols = vec_vols + edge_vec_vols
    s_list = Streamlines(s_list)

    return s_list, vec_vols


def fascicles_weight_by_mat(subj_folder,all_subj_folders, all_subj_names,i, ds, bundle_name):

    streamlines, lab_labels_index, affine, id, mat_weights, folder_name = load_vars(subj_folder,all_subj_folders, all_subj_names,i,bundle_name)

    streamline_dict = create_streamline_dict(streamlines, lab_labels_index, affine)

    s_list, vec_vols = weight_tracts_from_mat(id, streamline_dict, mat_weights)
    bundle_short = f'{bundle_name.split("_")[0]}_{bundle_name.split("_")[1]}'

    return folder_name, vec_vols, s_list, bundle_short

if __name__ == '__main__':
    i =7
    ds=1
    bundle_name = 'UF_L_mct001rt20'
    #condition = 'fa'
    condition = 'r_ax_fa_0.05'
    #condition = 'r_ax_fa_corr'
    #condition = 'ax'
    n=all_subj_names[i]
    s=all_subj_folders[i]
    folder_name, vec_vols, s_list, bundle_short = fascicles_weight_by_mat(subj_folder, all_subj_folders, all_subj_names, i, ds,bundle_name)
    #show_weighted_tractography(folder_name,vec_vols,s_list, bundle_short ,direction='left_side', downsamp=ds)


    img_name = rf'\{bundle_short}_{condition}.png'
    slices = [False, True, False]  # slices[0]-axial, slices[1]-saggital, slices[2]-coronal
    file_list = os.listdir(folder_name)

    for file in file_list:
        if r'_brain.nii' in file and file.startswith('r') and 'MPRAGE' in file:
            slice_file = nib.load(pjoin(folder_name, file))
            break

    bundlei = AdvanceInteractive(subj_folder, slice_file, s, n, img_name, bundle_name, slices)
    bundlei.scale = [-5,5]
    bundlei.load_bund()
    bundlei.s_list = s_list
    bundlei.load_vols()
    bundlei.vols = vec_vols
    print(min(vec_vols), max(vec_vols))

    bundlei.show_bundle_slices(color_map = 'r')


