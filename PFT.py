import numpy as np

from dipy.data import (read_stanford_labels, default_sphere,
                       read_stanford_pve_maps)
from dipy.direction import ProbabilisticDirectionGetter
from dipy.io.trackvis import save_trk
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.tracking.local import LocalTracking, ParticleFilteringTracking
from dipy.tracking import utils
from dipy.viz import window, actor, colormap as cmap


renderer = window.Renderer()

img_pve_csf, img_pve_gm, img_pve_wm = read_stanford_pve_maps()
hardi_img, gtab, labels_img = read_stanford_labels()

data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.affine
shape = labels.shape

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)
csd_fit = csd_model.fit(data, mask=img_pve_wm.get_data())

dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                               max_angle=20.,
                                               sphere=default_sphere)

from dipy.tracking.local import CmcTissueClassifier
from dipy.tracking.streamline import Streamlines

voxel_size = np.average(img_pve_wm.header()['pixdim'][1:4])
step_size = 0.2

cmc_classifier = CmcTissueClassifier.from_pve(img_pve_wm.get_data(),
                                              img_pve_gm.get_data(),
                                              img_pve_csf.get_data(),
                                              step_size=step_size,
                                              average_voxel_size=voxel_size)

# seeds are place in voxel of the corpus callosum containing only white matter
seed_mask = labels == 2
seed_mask[img_pve_wm.get_data() < 0.5] = 0
seeds = utils.seeds_from_mask(seed_mask, density=2, affine=affine)

# Particle Filtering Tractography
pft_streamline_generator = ParticleFilteringTracking(dg,
                                                     cmc_classifier,
                                                     seeds,
                                                     affine,
                                                     max_cross=1,
                                                     step_size=step_size,
                                                     maxlen=1000,
                                                     pft_back_tracking_dist=2,
                                                     pft_front_tracking_dist=1,
                                                     particle_count=15,
                                                     return_all=False)

# streamlines = list(pft_streamline_generator)
streamlines = Streamlines(pft_streamline_generator)
save_trk("pft_streamline.trk", streamlines, affine, shape)


renderer.clear()
renderer.add(actor.line(streamlines, cmap.line_colors(streamlines)))
window.record(renderer, out_path='pft_streamlines.png', size=(600, 600))