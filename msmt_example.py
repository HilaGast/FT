from dipy.denoise.localpca import mppca
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.data import default_sphere
from dipy.reconst.mcsd import (auto_response_msmt,
                               mask_for_response_msmt,
                               response_from_mask_msmt)
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response

from dipy.viz import window, actor
import numpy as np
import dipy.reconst.dti as dti
from dipy.data import get_fnames
from weighted_tracts import *
subj = all_subj_folders
names = all_subj_names

for s, n in zip(subj[0:1], names[0:1]):
    folder_name = subj_folder + s
    dir_name = folder_name + '\streamlines'
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
bval_file = bvec_file[:-4:] + 'bval'

sphere = default_sphere


bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
gtab = gradient_table(bvals, bvecs)

bvals = gtab.bvals
bvecs = gtab.bvecs

#Generate the mask for the data:
b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])

# denoising:
denoised_arr = mppca(data, mask=mask, patch_radius=2)

wm =r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep\questionnaire\YA_lab_Andrew_AhLi_20181129_0919\r20181129_091926T1wMPRAGERLs002a1001_brain_pve_2.nii'
wm = load_nifti(wm)[0]
gm =r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep\questionnaire\YA_lab_Andrew_AhLi_20181129_0919\r20181129_091926T1wMPRAGERLs002a1001_brain_pve_1.nii'
gm=load_nifti(gm)[0]
csf =r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep\questionnaire\YA_lab_Andrew_AhLi_20181129_0919\r20181129_091926T1wMPRAGERLs002a1001_brain_pve_0.nii'
csf=load_nifti(csf)[0]

mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab, data, roi_radii=10,
                                                    wm_fa_thr=0.7,
                                                    gm_fa_thr=0.3,
                                                    csf_fa_thr=0.15,
                                                    gm_md_thr=0.001,
                                                    csf_md_thr=0.0032)
mask_wm=mask_wm.astype(float)
mask_gm=mask_gm.astype(float)
mask_csf=mask_csf.astype(float)

mask_wm *= wm
mask_gm *= gm
mask_csf *= csf


response_wm, response_gm, response_csf = response_from_mask_msmt(gtab, data,
                                                                 mask_wm,
                                                                 mask_gm,
                                                                 mask_csf)

#auto_response_wm, auto_response_gm, auto_response_csf = auto_response_msmt(gtab, data, roi_radii=10)
ubvals = unique_bvals_tolerance(bvals)
response_mcsd = multi_shell_fiber_response(sh_order=8, bvals=ubvals,
                                           wm_rf=response_wm,
                                           csf_rf=response_csf,
                                           gm_rf=response_gm)


mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)
mcsd_fit = mcsd_model.fit(denoised_arr)

#mcsd_odf = mcsd_fit.odf(sphere)
coeff = mcsd_model.fitter(denoised_arr)

def show_odf_slicer(mcsd_model,data,slice):
    mcsd_fit = mcsd_model.fit(data[:, :, slice])
    mcsd_odf = mcsd_fit.odf(sphere)
    fodf_spheres = actor.odf_slicer(mcsd_odf, sphere=sphere, scale=0.01,
                                norm=False, colormap='plasma')

    interactive = False
    ren = window.Renderer()
    ren.add(fodf_spheres)
    ren.reset_camera_tight()

    print('Saving illustration as msdodf.png')
    window.record(ren, out_path='msdodf.png', size=(600, 600))

    if interactive:
        window.show(ren)