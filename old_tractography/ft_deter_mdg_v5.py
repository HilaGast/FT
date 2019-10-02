# from dipy.data import read_stanford_labels
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
import nibabel as nib
import os
from dipy.core.gradients import gradient_table
import numpy as np

subj_name = "subj9"
path_name = r'C:\Users\Admin\my_scripts\NaYa_subj9'

file_name = r'20190211_134016ep2dd155D60MB3APs004a001.nii'
dwi_file_name = os.path.join(path_name, file_name)
hardi_img = nib.load(dwi_file_name)

bval_name = r'20190211_134016ep2dd155D60MB3APs004a001.bval'
bvec_name = r'20190211_134016ep2dd155D60MB3APs004a001.bvec'
bval_file = os.path.join(path_name, bval_name)
bvec_file = os.path.join(path_name, bvec_name)
gtab = gradient_table(bval_file, bvec_file, small_delta=15.5)

label_name = r'r20190211_134016T1wMPRAGERLs002a1001_brain_seg.nii'
labels_file_name = os.path.join(path_name, label_name)
labels_img = nib.load(labels_file_name)
labels = labels_img.get_data()

data = hardi_img.get_data()
affine = hardi_img.affine

cc_file_name = r'cc_mask.nii'
cc_file = os.path.join(path_name, cc_file_name)
cc_img = nib.load(cc_file)
cc_mask = cc_img.get_data()

''' white_matter - the entire brain mask to track fibers
seed_mask - the mask which determine where to start tracking'''
white_matter = (labels == 3) | (labels == 2)  # 3-WM, 2-GM
seed_mask = white_matter
#seed_mask = cc_mask == 1
# white_matter = (seed_mask)

seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)

csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=8)

csd_fit = csd_model.fit(data, mask=white_matter)

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy

tensor_model = dti.TensorModel(gtab)
tenfit = tensor_model.fit(data, mask=white_matter)

FA = fractional_anisotropy(tenfit.evals)
classifier = ThresholdTissueClassifier(FA, .2)

from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.streamline import save_trk, load_trk

detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                             max_angle=40.,
                                                             sphere=default_sphere)
from dipy.tracking.streamline import Streamlines

streamlines = Streamlines(LocalTracking(detmax_dg, classifier, seeds, affine, step_size=.1))

long_streamlines = np.ones((len(streamlines)),bool)
for i in range(0,len(streamlines)):
    if streamlines[i].shape[0] < 70:
        long_streamlines[i] = False
streamlines = streamlines[long_streamlines]


from dipy.viz import window, actor, colormap as cmap

streamlines_actor = actor.line(streamlines, cmap.line_colors(streamlines))

# weighted streamlines:
from dipy.tracking.streamline import transform_streamlines
''' FA weighting:
fa_name = r'20190211_134016ep2dd155D60MB3APs004a001_FA.nii'
fa_file_name = os.path.join(path_name,fa_name)
fa_img = nib.load(fa_file_name)
fa = fa_img.get_data()
affine = fa_img.get_affine()
streamlines_native = transform_streamlines(streamlines, np.linalg.inv(affine))
streamlines_actor2 = actor.line(streamlines_native, fa, linewidth=0.1) '''
''' PASI weighting:'''
pasi_name = r'20190211_134016ep2dd155D60MB3APs004a001_pasiS.nii'
pasi_file_name = os.path.join(path_name,pasi_name)
pasi_img = nib.load(pasi_file_name)
pasi = pasi_img.get_data()
affine = pasi_img.get_affine()
streamlines_native = transform_streamlines(streamlines, np.linalg.inv(affine))
hue = (0.0, 1.0)  #
saturation = (0.0, 1.0)  # white to black
scale = (0,9)
lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                       saturation_range=saturation, scale_range = scale)
streamlines_actor2 = actor.line(streamlines_native, pasi, linewidth=0.1,lookup_colormap=lut_cmap)
bar = actor.scalar_bar(lut_cmap)
#bar = actor.scalar_bar()

r = window.Renderer()
r.add(streamlines_actor2)
r.add(bar)
# window.show(renderer, size=(600, 600), reset_camera=False)
window.record(r, out_path='bundle2.png', size=(600, 600))
window.show(r)



'''
# Create the 3D display.
r = window.Renderer()
r.add(streamlines_actor)

# Save still images for this static example.
window.record(r, n_frames=1, out_path='deterministic.png', size=(800, 800))
window.show(r)


## load trk file:
def load_ft():
    from dipy.io.streamline import load_trk
    from dipy.viz import window, actor, colormap as cmap

    streams, hdr = load_trk(tract_name)
    streamlines = Streamlines(streams)

    streamlines_actor = actor.line(streamlines, cmap.line_colors(streamlines))

    # Create the 3D display.
    r = window.Renderer()
    r.add(streamlines_actor)

# Save still images for this static example.
# window.record(r, n_frames=1, out_path='deterministic.png', size=(800, 800))
# window.show(r)
'''