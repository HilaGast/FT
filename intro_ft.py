#from dipy.data import read_stanford_labels #loading
from dipy.tracking import utils #a tool for affine transformation
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
from dipy.io.streamline import save_trk
from dipy.viz import window, actor, colormap as cmap
import nibabel as nib
import os
from dipy.core.gradients import gradient_table

from dipy.reconst.shm import CsaOdfModel
from dipy.direction import BootDirectionGetter
from dipy.tracking.streamline import Streamlines
from dipy.data import default_sphere
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)

from dipy.direction import ProbabilisticDirectionGetter


renderer = window.Renderer()

#hardi_img, gtab, labels_img = read_stanford_labels() #loads example data
# hardi_img: a 4D nifty , class - 'nibabel.nifti1.Nifti1Image'
# gtab: gradient data, a GradientTable object
# labels_img: a 3D nifty with dims hardi_img.size(0,1,2) for ROI to fiber tracking seeds.
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

csa_model = CsaOdfModel(gtab, sh_order=6)
gfa = csa_model.fit(data, mask=white_matter).gfa
classifier = ThresholdTissueClassifier(gfa, .1)



boot_dg_csd = BootDirectionGetter.from_data(data, csd_model, max_angle=15.,
                                            sphere=default_sphere)
boot_streamline_generator = LocalTracking(boot_dg_csd, classifier, seeds,
                                         affine, step_size=.5)
streamlines = Streamlines(boot_streamline_generator)

list_streamlines=list(streamlines)
for i in range(0,len(list_streamlines)):
    if list_streamlines[i].shape[0] < 60:
        list_streamlines[i]=[]
list_streamlines= [tract for tract in list_streamlines if tract!=[]]
streamlines = Streamlines(list_streamlines)


renderer.clear()
renderer.add(actor.line(streamlines, cmap.line_colors(streamlines)))
window.record(renderer, out_path='bootstrap_dg_CSD.png', size=(600, 600))

streamlines_actor = actor.line(streamlines, cmap.line_colors(streamlines))

    # Create the 3D display.
r = window.Renderer()
r.add(streamlines_actor)

# Save still images for this static example.
window.record(r, n_frames=1, out_path='bootstrap_dg_CSD.png', size=(800, 800))
window.show(r)




    ######################################################################################################################

response, ratio = auto_response(gtab, data, roi_radius=1, fa_thr=0.1)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)


prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                    max_angle=30.,
                                                    sphere=default_sphere)

import numpy as np
streamlines_generator = LocalTracking(prob_dg, classifier, seeds,
                                      affine=np.eye(4), step_size=.5,
                                      max_cross=3)

# Generate streamlines object.
streamlines = Streamlines(streamlines_generator)

streamlines_actor = actor.line(streamlines, cmap.line_colors(streamlines))

    # Create the 3D display.
r = window.Renderer()
r.add(streamlines_actor)

# Save still images for this static example.
window.record(r, n_frames=1, out_path='probabilistic.png', size=(800, 800))
window.show(r)
