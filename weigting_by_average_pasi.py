from dipy.tracking.streamline import values_from_volume, unlist_streamlines
from dipy.io.streamline import load_trk
from dipy.tracking.streamline import Streamlines
import os
import nibabel as nib
import numpy as np
from dipy.viz import window, actor, colormap as cmap
from dipy.tracking.streamline import transform_streamlines

subj = r'\BeEf_subj7'
stream_file = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5' +subj+ '\streamlines' +subj+'.trk'
streams, hdr = load_trk(stream_file)
streamlines = Streamlines(streams)
stream = list(s1)
weight_by = 'pasiS'
folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5'+subj
file_list = os.listdir(folder_name)
for file in file_list:
    if weight_by in file and file.endswith('.nii'):
        weight_by_file = os.path.join(folder_name, file)
        weight_by_img = nib.load(weight_by_file)
        weight_by_data = weight_by_img.get_data()
        data_affine = weight_by_img.get_affine()
A = values_from_volume(weight_by_data, stream, affine=data_affine)
pasi_stream = []
for i,s in enumerate(A):
    pasi_stream.append(np.mean(s))
scale=[0,100]
saturation = [0.0,1.0]
hue = [0,1]

lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                       saturation_range=saturation, scale_range=scale)
streamlines_actor = actor.line(streamlines, pasi_stream, linewidth=0.1, lookup_colormap=lut_cmap)
bar = actor.scalar_bar(lut_cmap)
r = window.Renderer()
r.add(streamlines_actor)
r.add(bar)
window.show(r)