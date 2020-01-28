from __future__ import division

import os
import nibabel as nib
from dipy.viz import window, actor, ui


fname_t1 =r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep\YA_lab_Andrew_AhLi_20181129_0919\20181129_091926ep2dd155D60MB3APs005a001_1.5_2_AxPasi5.nii'


img = nib.load(fname_t1)
data = img.get_data()
affine = img.affine

renderer = window.Renderer()
renderer.background((0.5, 0.5, 0.5))

mean, std = data[data > 0].mean(), data[data > 0].std()
value_range = (mean - 0.5 * std, mean + 1.5 * std)

slice_actor = actor.slicer(data, affine, value_range)
slice_actor.display(slice_actor.shape[0]//2, None, None)
renderer.add(slice_actor)

slice_actor2 = slice_actor.copy()

slice_actor2.display(slice_actor2.shape[0]//2, None, None)

#renderer.add(slice_actor2)

renderer.reset_camera()
renderer.zoom(1.4)

window.show(renderer, size=(600, 600), reset_camera=False)

window.record(renderer, out_path=r'C:\Users\Admin\Desktop\preliminary results\pasi_midsag.png', size=(600, 600), reset_camera=False)