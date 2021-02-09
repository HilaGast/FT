import numpy as np
from dipy.viz import actor, window, ui
import nibabel as nib
from single_fascicle_vizualization import *
from weighted_tracts import weighting_streamlines,load_dwi_files,load_weight_by_img
from clustering.cluster_fascicles import *
from os.path import join as pjoin
from dipy.io.streamline import load_trk


def load_vars(volume_file, main_folder,s,n,img_name = r'' ):
    folder_name = main_folder + s
    s_img = folder_name + r'\streamlines' + img_name
    mask_type='cc'
    scale = [3, 11]
    hue = [0.25, -0.05]  # Hot
    saturation = [0.1, 1.0]
    axial = False
    saggital = True
    coronal = False
    volume = volume_file.get_data()
    shape = volume.shape
    affine = volume_file.affine
    nii_file = load_dwi_files(folder_name)[5]
    s_list, vec_vols = streamline_mean_fascicle_value_weighted(folder_name, n, nii_file,mask_type)
    return folder_name, s_img, scale, hue, saturation, axial, saggital, coronal, volume, shape, affine, s_list, vec_vols


def register_coords(world_coords, streamlines, affine, volume):
    """If we want to see the objects in native space we need to make sure that all objects which are currently
    in world coordinates are transformed back to native space using the inverse of the affine."""
    if not world_coords:
        from dipy.tracking.streamline import transform_streamlines
        streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))

    if not world_coords:
        image_actor_z = actor.slicer(volume, affine=np.eye(4))
    else:
        image_actor_z = actor.slicer(volume, affine)

    return streamlines, image_actor_z


def create_slicers(image_actor_z, shape, slicer_opacity):
    image_actor_z.opacity(slicer_opacity)

    image_actor_x = image_actor_z.copy()
    x_midpoint = int(np.round(shape[0] / 2))
    image_actor_x.display_extent(x_midpoint,
                                 x_midpoint, 0,
                                 shape[1] - 1,
                                 0,
                                 shape[2] - 1)

    image_actor_y = image_actor_z.copy()
    y_midpoint = int(np.round(shape[1] / 2))
    image_actor_y.display_extent(0,
                                 shape[0] - 1,
                                 y_midpoint,
                                 y_midpoint,
                                 0,
                                 shape[2] - 1)

    return image_actor_z, image_actor_x, image_actor_y


class AdvanceInteractive:

    def __init__(self,main_folder, slices_file, s , n , img_name, bundle,slices):
        self.main_folder = main_folder
        self.subj_folder = s
        self.subj_name = n
        self.img_name = img_name
        self.bundle = bundle
        self.axial = slices[0]
        self.saggital = slices[1]
        self.coronal = slices[2]
        self.folder_name = main_folder + s
        self.s_img = self.folder_name + r'\streamlines' + img_name
        self.scale = [3, 10]
        self.hue = [0.25, -0.05]  # Hot
        self.saturation = [0.1, 1.0]
        self.slices = slices_file.get_data()
        self.shape = self.slices.shape
        #self.affine = slices_file.affine
        _, _, self.affine, _, _, self.nii_file, self.bvec_file = load_dwi_files(self.folder_name)

    def load_bund(self):
        file_list = os.listdir(self.folder_name + r'\streamlines')
        for file in file_list:
            if self.bundle in file and '.trk' in file:
                self.bundle_file = pjoin(self.folder_name + r'\streamlines', file)
                s_list = load_trk(self.bundle_file,"same", bbox_valid_check=False)
                self.s_list = s_list.streamlines
                break

    def load_vols(self, weight_by='_1.5_2_AxPasi5'):
        self.vols = weighting_streamlines(self.folder_name,self.s_list,self.bvec_file,weight_by=weight_by)

    def show_bundle_slices(self, color_map = 'ax'):

        world_coords = True
        streamlines, image_actor_z = register_coords(world_coords, self.s_list, self.affine, self.slices)

        if color_map == 'ax':
            lut_cmap = actor.colormap_lookup_table(hue_range=self.hue,
                                                   saturation_range=self.saturation, scale_range=self.scale)
            bar = actor.scalar_bar(lut_cmap)
            stream_actor = actor.line(self.s_list, self.vols, linewidth=1.2, lookup_colormap=lut_cmap)
            ren = window.Scene()
            ren.add(bar)

        elif color_map == 'fa':
            self.vols = list(np.asarray(self.vols)/100)
            lut_cmap = actor.colormap_lookup_table(hue_range=self.hue,
                                                   saturation_range=self.saturation, scale_range=self.scale)
            bar = actor.scalar_bar(lut_cmap)
            stream_actor = actor.line(self.s_list, self.vols, linewidth=1.2, lookup_colormap=lut_cmap)
            ren = window.Scene()
            ren.add(bar)

        elif color_map == 'r':
            from fury.colormap import create_colormap
            self.vols.append(1)
            self.vols.append(-1)
            cmap = create_colormap(np.asarray(self.vols), name='seismic')
            self.vols = self.vols[:-2]
            cmap = cmap[:-2]
            stream_actor = actor.line(self.s_list, cmap, linewidth=1.2)
            ren = window.Scene()

        slicer_opacity = 0.6
        image_actor_z, image_actor_x, image_actor_y = create_slicers(image_actor_z, self.shape, slicer_opacity)

        ren.add(stream_actor)

        if self.axial:
            ren.add(image_actor_z)  # axial
        if self.saggital:
            ren.add(image_actor_x)  # saggital
        if self.coronal:
            ren.add(image_actor_y)  # coronal

        show_m = window.ShowManager(ren, size=(1200, 900))
        show_m.initialize()

        def sliders(shape, slicer_opacity):
            line_slider_z = ui.LineSlider2D(min_value=0,
                                            max_value=shape[2] - 1,
                                            initial_value=shape[2] / 2,
                                            text_template="{value:.0f}",
                                            length=140)

            line_slider_x = ui.LineSlider2D(min_value=0,
                                            max_value=shape[0] - 1,
                                            initial_value=shape[0] / 2,
                                            text_template="{value:.0f}",
                                            length=140)

            line_slider_y = ui.LineSlider2D(min_value=0,
                                            max_value=shape[1] - 1,
                                            initial_value=shape[1] / 2,
                                            text_template="{value:.0f}",
                                            length=140)

            opacity_slider = ui.LineSlider2D(min_value=0.0,
                                             max_value=1.0,
                                             initial_value=slicer_opacity,
                                             length=140)
            return line_slider_x, line_slider_y, line_slider_z, opacity_slider

        def build_label(text):
            label = ui.TextBlock2D()
            label.message = text
            label.font_size = 13
            label.font_family = 'Arial'
            label.justification = 'left'
            label.bold = False
            label.italic = False
            label.shadow = False
            label.background = (0, 0, 0)
            label.color = (1, 1, 1)

            return label

        line_slider_x, line_slider_y, line_slider_z, opacity_slider = sliders(self.shape, slicer_opacity)

        def change_slice_z(slider):
            z = int(np.round(slider.value))
            image_actor_z.display_extent(0, self.shape[0] - 1, 0, self.shape[1] - 1, z, z)

        def change_slice_x(slider):
            x = int(np.round(slider.value))
            image_actor_x.display_extent(x, x, 0, self.shape[1] - 1, 0, self.shape[2] - 1)

        def change_slice_y(slider):
            y = int(np.round(slider.value))
            image_actor_y.display_extent(0, self.shape[0] - 1, y, y, 0, self.shape[2] - 1)

        def change_opacity(slider):
            slicer_opacity = slider.value
            image_actor_z.opacity(slicer_opacity)
            image_actor_x.opacity(slicer_opacity)
            image_actor_y.opacity(slicer_opacity)

        line_slider_z.on_change = change_slice_z
        line_slider_x.on_change = change_slice_x
        line_slider_y.on_change = change_slice_y
        opacity_slider.on_change = change_opacity

        line_slider_label_z = build_label(text="Z Slice")
        line_slider_label_x = build_label(text="X Slice")
        line_slider_label_y = build_label(text="Y Slice")
        opacity_slider_label = build_label(text="Opacity")

        panel = ui.Panel2D(size=(300, 200),
                           color=(1, 1, 1),
                           opacity=0.1,
                           align="right")
        panel.center = (1030, 120)

        panel.add_element(line_slider_label_x, (0.1, 0.75))
        panel.add_element(line_slider_x, (0.38, 0.75))
        panel.add_element(line_slider_label_y, (0.1, 0.55))
        panel.add_element(line_slider_y, (0.38, 0.55))
        panel.add_element(line_slider_label_z, (0.1, 0.35))
        panel.add_element(line_slider_z, (0.38, 0.35))
        panel.add_element(opacity_slider_label, (0.1, 0.15))
        panel.add_element(opacity_slider, (0.38, 0.15))

        ren.add(panel)

        global size
        size = ren.GetSize()

        show_m.initialize()

        interactive = True

        ren.zoom(1.1)
        ren.reset_clipping_range()

        def win_callback(obj, event):
            global size
            if size != obj.GetSize():
                size_old = size
                size = obj.GetSize()
                size_change = [size[0] - size_old[0], 0]
                panel.re_align(size_change)

        if interactive:
            show_m.add_window_callback(win_callback)
            show_m.render()
            show_m.start()

            ren.set_camera(ren.camera_info())
            window.record(ren, out_path=self.s_img, size=(1000, 1000))

        del show_m


if __name__ == '__main__':

    main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep\questionnaire'
    s = all_subj_folders[0]
    n = all_subj_names[0]
    img_name = r'\fascicles_AxCaliber7_weighted_4d_cr_pasivals.png'
    bundle = r'cr'
    #bundle = r'wholebrain_1d_labmask'
    slices = [False, True, False] #slices[0]-axial, slices[1]-saggital, slices[2]-coronal
    file_list = os.listdir(main_folder + s)

    for file in file_list:
        #if r'_highres.nii' in file and file.startswith('r') and 'Labels' in file:
        if r'_brain.nii' in file and file.startswith('r') and 'MPRAGE' in file:
        #if r'FA' in file and file.endswith('.nii'):
            slice_file = nib.load(pjoin(main_folder+s, file))
            break

    bundlei = AdvanceInteractive(main_folder,slice_file, s, n, img_name, bundle, slices)
    bundlei.load_bund()
    bundlei.load_vols(weight_by='_2_2_AxPasi7')
    bundlei.show_bundle_slices()

    method = 'mam'
    tracts_num = bundlei.s_list.__len__()
    X = clustering_input(method,tracts_num,bundlei.s_list,bundlei.vols)
    methods = ['agglomerative','kmeans']
    method = methods[1]
    g=[2,3,4,5,6]
    for i in g:
        model = compute_clustering_model(method,X,i)
        weighted_clusters(model, bundlei.s_list, bundlei.vols,bundlei.folder_name, file_name = 'clustered_'+bundle+'_'+str(i))

