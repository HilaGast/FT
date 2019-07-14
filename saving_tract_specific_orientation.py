



tract_path = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5\NaYa_subj9\streamlines\NaYa_subj9.trk'
from dipy.io.streamline import load_trk
streams, hdr = load_trk(tract_path)
streamlines = Streamlines(streams)

from dipy.viz import window, actor, colormap as cmap
streamlines_actor = actor.line(streamlines, cmap.line_colors(streamlines))
r = window.Renderer()
r.add(streamlines_actor)
window.show(r)
r.set_camera(r.camera_info())
save_as = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5\NaYa_subj9\streamlines\deterministic.png'
window.record(r, n_frames=1, out_path=save_as, size=(800, 800))

'''
from dipy.tracking.streamline import transform_streamlines
hue = [0.0,1.0]
saturation = [0.0,1.0]
scale = [0,9]
pasi_file  = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5\BeEf_subj7\20190131_132019ep2dd155D60MB3APs005a001_pasiS.nii'
weight_by_file = pasi_file
weight_by_img = nib.load(weight_by_file)
weight_by_data = weight_by_img.get_data()
affine = weight_by_img.get_affine()
streamlines_native = transform_streamlines(streamlines, np.linalg.inv(affine))
lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                       saturation_range=saturation, scale_range=scale)
streamlines_actor = actor.line(streamlines_native, weight_by_data, linewidth=0.1, lookup_colormap=lut_cmap)
bar = actor.scalar_bar(lut_cmap)

ren = window.Renderer()
ren.add(streamlines_actor)
ren.add(bar)
window.show(ren)
window.record(r, out_path='bundle2.png', size=(800, 800))
'''