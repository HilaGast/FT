#from dipy.data import read_stanford_labels
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
import nibabel as nib
import os
from dipy.core.gradients import gradient_table
import numpy as np
subj_name = "C143"
charmed_path_name = r'C:\Users\Admin\my_scripts\C1156_43\YA_lab_Assi_GaHi_20171231_1320\19_CHARMED1_ep2d_advdiff_64_optimized_d18D38'
path_name=r'C:\Users\Admin\my_scripts\C1156_43\YA_lab_Assi_GaHi_20171231_1320\23_AxCaliber3D1_ep2d_advdiff_30dir_b3000_d11.3D60'

file_name=r'20171231_132011CHARMED1ep2dadvdiff64optimizedd18D38s019a001.nii'
charmed_file_name = os.path.join(charmed_path_name,file_name)
hardi_img = nib.load(charmed_file_name)

bval_name = r'20171231_132011CHARMED1ep2dadvdiff64optimizedd18D38s019a001A.bval'
bvec_name = r'20171231_132011CHARMED1ep2dadvdiff64optimizedd18D38s019a001A.bvec'
bval_file = os.path.join(charmed_path_name,bval_name)
bvec_file = os.path.join(charmed_path_name,bvec_name)
gtab = gradient_table(bval_file,bvec_file)

bvl=list(gtab.bvals)
bvl[:-3:-1] = bvl[1:3:1]
bvl[1:3:1] = [0.0,0.0]
gtab.bvals = np.asarray(bvl)
bvc=list(gtab.bvecs)
bvc[:-3:-1] = bvc[1:3:1]
bvc[1:3:1] = [np.asarray([0.0,0.0,0.0]),np.asarray([0.0,0.0,0.0])]
gtab.bvecs = np.asarray(bvc)

label_name=r'r20171231_132011MPRAGEEnchancedContrasts002a1001_brain_seg.nii'
labels_file_name = os.path.join(charmed_path_name,label_name)
labels_img = nib.load(labels_file_name)

data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.affine


seed_mask = labels == 3
white_matter = (seed_mask)
seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)




csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
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
                                                             max_angle=30.,
                                                             sphere=default_sphere)
from dipy.tracking.streamline import Streamlines

streamlines = Streamlines(LocalTracking(detmax_dg, classifier, seeds, affine, step_size=.5))

long_streamlines = np.ones((len(streamlines)),bool)
for i in range(0,len(streamlines)):
    if streamlines[i].shape[0] < 70:
        long_streamlines[i] = False
streamlines = streamlines[long_streamlines]

dir_name = os.path.join(r"C:\Users\Admin\Miniconda3\envs\LabPy\FT\FT\ft_deter_mdg")
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
tract_name = os.path.join(dir_name,(subj_name+".trk"))
save_trk(tract_name, streamlines, affine,
         labels.shape)

from dipy.viz import window, actor, colormap as cmap

streamlines_actor = actor.line(streamlines, cmap.line_colors(streamlines))

    # Create the 3D display.
r = window.Renderer()
r.add(streamlines_actor)

# Save still images for this static example.
window.record(r, n_frames=1, out_path='probabilistic.png', size=(800, 800))
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
window.record(r, n_frames=1, out_path='probabilistic.png', size=(800, 800))
window.show(r)