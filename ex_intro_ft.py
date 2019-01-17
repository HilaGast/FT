from dipy.data import read_stanford_labels #loading
from dipy.tracking import utils #a tool for affine transformation
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
from dipy.io.streamline import save_trk
from dipy.viz import window, actor, colormap as cmap
import nibabel as nib
import os
from dipy.core.gradients import gradient_table

renderer = window.Renderer()

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

#hardi_img, gtab, labels_img = read_stanford_labels() #loads example data
# hardi_img: a 4D nifty , class - 'nibabel.nifti1.Nifti1Image'
# gtab: gradient data, a GradientTable object
# labels_img: a 3D nifty with dims hardi_img.size(0,1,2) for ROI to fiber tracking seeds.
path_name=r'C:\Users\Admin\my_scripts\C1156_43\YA_lab_Assi_GaHi_20171231_1320\23_AxCaliber3D1_ep2d_advdiff_30dir_b3000_d11.3D60'

file_name=r'20171231_132011AxCaliber3D1ep2dadvdiff30dirb3000d113D60s023a001.nii'
charmed_file_name = os.path.join(path_name,file_name)
hardi_img = nib.load(charmed_file_name)

bval_name = r'20171231_132011AxCaliber3D1ep2dadvdiff30dirb3000d113D60s023a001.bval'
bvec_name = r'20171231_132011AxCaliber3D1ep2dadvdiff30dirb3000d113D60s023a001.bvec'
bval_file = os.path.join(path_name,bval_name)
bvec_file = os.path.join(path_name,bvec_name)
gtab = gradient_table(bval_file,bvec_file)




data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.affine


seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)

csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)

from dipy.reconst.shm import CsaOdfModel
csa_model = CsaOdfModel(gtab, sh_order=6)
gfa = csa_model.fit(data, mask=white_matter).gfa
classifier = ThresholdTissueClassifier(gfa, .25)

from dipy.direction import BootDirectionGetter
from dipy.tracking.streamline import Streamlines
from dipy.data import small_sphere

boot_dg_csd = BootDirectionGetter.from_data(data, csd_model, max_angle=30.,
                                            sphere=small_sphere)
boot_streamline_generator = LocalTracking(boot_dg_csd, classifier, seeds,
                                          affine, step_size=.5)
streamlines = Streamlines(boot_streamline_generator)

renderer.clear()
renderer.add(actor.line(streamlines, cmap.line_colors(streamlines)))
window.record(renderer, out_path='bootstrap_dg_CSD.png', size=(600, 600))

