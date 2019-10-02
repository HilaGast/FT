

from dipy.viz import window, actor, colormap as cmap
from dipy.tracking.streamline import transform_streamlines, values_from_volume
import numpy as np
from FT.weighted_tracts import load_ft, load_dwi_files
from FT.all_subj import all_subj_names
import nibabel as nib

subj = all_subj_names[6:]

for s in subj:
    main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5' + s
    tract_path = main_folder + r"\streamlines" + s + '_genu_cleaned.trk'
    streamlines = load_ft(tract_path)
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(main_folder)

    streamlines_native = transform_streamlines(streamlines, np.linalg.inv(affine))
    weight_by = 'pasiS'
    hue = [0.0,1.0]
    saturation = [0.0,1.0]
    scale = [0,100]


    weight_by_file = bvec_file[:-5:] + '_' + weight_by + '.nii'
    weight_by_img = nib.load(weight_by_file)
    weight_by_data = weight_by_img.get_data()
    affine = weight_by_img.affine
    stream = list(streamlines)
    vol_per_tract = values_from_volume(weight_by_data, stream, affine=affine)
    pfr_file = bvec_file[:-5:] + '_pfrS.nii'
    pfr_img = nib.load(pfr_file)
    pfr_data = pfr_img.get_data()



    lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                       saturation_range=saturation, scale_range=scale)
    streamlines_actor = actor.line(streamlines_native, pfr_data, lookup_colormap=lut_cmap)
    bar = actor.scalar_bar()

    r = window.Renderer()
    r.add(streamlines_actor)
    r.add(bar)
    window.show(r)
    fig_path = main_folder + r"\streamlines" + s + '_genu_dist.png'
    r.set_camera(r.camera_info())
    window.record(r, path_numbering=True, n_frames=3, az_ang=30, out_path=fig_path, size=(800, 800))