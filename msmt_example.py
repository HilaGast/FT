from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.data import default_sphere
from dipy.reconst.mcsd import (response_from_mask_msmt)
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response, MSDeconvFit
from dipy.viz import window, actor
import numpy as np
from weighted_tracts import *
subj = all_subj_folders
names = all_subj_names

for s, n in zip(subj[9:10], names[9:10]):
    folder_name = subj_folder + s
    dir_name = folder_name + '\streamlines'
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
bval_file = bvec_file[:-4:] + 'bval'

sphere = default_sphere


bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
bvals = np.around(bvals, decimals=-2)
gtab = gradient_table(bvals, bvecs)

bvals = gtab.bvals
bvecs = gtab.bvecs

denoised_arr = data

tissue_mask =r'F:\Hila\Ax3D_Pack\V6\after_file_prep\YA_lab_Yaniv_002334_20210107_1820\r20210107_182004T1wMPRAGERLs008a1001_brain_seg.nii'
t_mask_img = load_nifti(tissue_mask)[0]
wm = t_mask_img==3
gm = t_mask_img==2
csf = t_mask_img==1

mask_wm=wm.astype(float)
mask_gm=gm.astype(float)
mask_csf= csf.astype(float)

response_wm, response_gm, response_csf = response_from_mask_msmt(gtab, data,
                                                                 mask_wm,
                                                                 mask_gm,
                                                                 mask_csf)

ubvals = unique_bvals_tolerance(bvals)
response_mcsd = multi_shell_fiber_response(sh_order=8, bvals=ubvals,
                                           wm_rf=response_wm,
                                           csf_rf=response_csf,
                                           gm_rf=response_gm)


mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)
mcsd_fit = mcsd_model.fit(denoised_arr)
sh_coeff = mcsd_fit.all_shm_coeff
nan_count = len(np.argwhere(np.isnan(sh_coeff[..., 0])))
coeff = mcsd_fit.all_shm_coeff
n_vox = coeff.shape[0]*coeff.shape[1]*coeff.shape[2]
print(f'{nan_count/n_vox} of the voxels did not complete fodf calculation, NaN values replaced with 0')
coeff = np.where(np.isnan(coeff), 0, coeff)
mcsd_fit = MSDeconvFit(mcsd_model,coeff,None)
np.save(folder_name+r'\coeff.npy',coeff)

gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
fa, classifier = create_fa_classifier(gtab, data, white_matter)
lab_labels_index = nodes_by_index_general(folder_name,atlas='yeo7_200')[0]
seeds = create_seeds(folder_name, lab_labels_index, affine, use_mask=False, mask_type='cc', den=5)
del(coeff,data,response_wm,response_gm,response_csf,response_mcsd,denoised_arr,wm,gm,csf,t_mask_img,sh_coeff,mask_csf,mask_wm,mask_gm)

streamlines = create_streamlines(mcsd_fit, classifier, seeds, affine)
save_ft(folder_name, n, streamlines, nii_file, file_name="_wholebrain_5d_labmask_msmt.trk")
#weighting_streamlines(folder_name,streamlines,bvec_file,show=True,scale=[3,12],hue=[0.25, -0.05],saturation=[0.1,1],fig_type='_cr_reco_msmt_4d',weight_by='3_2_AxPasi7')

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