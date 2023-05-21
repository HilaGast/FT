import os
import nibabel as nib


def show_tracts_simple(s_list, folder_name, fig_type, time2present=1, down_samp=1, vec_vols=None,hue=[0.25, -0.05],saturation=[0.1,1],scale=[3, 6], weighted=False, colormap=None, min=0, max=1):
    from dipy.viz import window, actor
    from fury.colormap import create_colormap
    import numpy as np
    if weighted:
        if down_samp != 1:
            vec_vols = vec_vols[::down_samp]
            s_list = s_list[::down_samp]
        r = window.Scene()
        if colormap:
            vec_vols.append(max)
            vec_vols.append(min)
            cmap = create_colormap(np.asarray(vec_vols), name='seismic')
            vec_vols = vec_vols[:-2]
            cmap = cmap[:-2]
            streamlines_actor = actor.line(s_list, cmap, linewidth=2)

        else:
            cmap = actor.colormap_lookup_table(hue_range=hue, saturation_range=saturation, scale_range=scale)
            streamlines_actor = actor.line(s_list, vec_vols, linewidth=2, lookup_colormap=cmap)
            bar = actor.scalar_bar(cmap)
            r.add(bar)
        r.add(streamlines_actor)
        for i in range(time2present):
            weighted_img = f'{folder_name}{os.sep}streamlines{os.sep}{fig_type}_{str(i+1)}.png'
            window.show(r)
            r.set_camera(r.camera_info())
            window.record(r, out_path=weighted_img, size=(800, 800))
    else:
        if down_samp != 1:
            s_list = s_list[::down_samp]
        lut_cmap = actor.colormap_lookup_table(hue_range=hue, saturation_range=saturation, scale_range=scale)
        streamlines_actor = actor.line(s_list, linewidth=2, lookup_colormap=lut_cmap)
        r = window.Scene()
        r.add(streamlines_actor)
        for i in range(time2present):
            non_weighted_img = f'{folder_name}{os.sep}streamlines{os.sep}non_weighted_{fig_type}_{str(i+1)}.png'
            window.show(r)
            r.set_camera(r.camera_info())
            window.record(r, out_path=non_weighted_img, size=(800, 800))


def show_tracts_by_mask(folder_name, mask_file_name, s_list, affine,fig_type=None, downsamp=1):
    from dipy.tracking import utils
    from Tractography.files_saving import save_ft
    from dipy.tracking.streamline import Streamlines


    mask_file = os.path.join(folder_name, mask_file_name+'.nii')
    mask_img = nib.load(mask_file).get_fdata()
    mask_include = mask_img > 0
    masked_streamlines = utils.target(s_list, affine, mask_include)
    masked_streamlines = Streamlines(masked_streamlines)

    save_ft(folder_name, masked_streamlines, mask_file, file_name = f"{mask_file_name}.trk")
    if not fig_type:
        fig_type = mask_file_name
    show_tracts_simple(masked_streamlines, folder_name, fig_type)



