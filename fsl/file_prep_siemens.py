from fsl.file_prep import (
    os_path_2_fsl,
    create_inv_mat,
    flirt_primary_guess,
    fnirt_from_atlas_2_subj,
    apply_fnirt_warp_on_label,
)
import os
from dipy.align import motion_correction
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma


def subj_files(exp):
    diffusion_file = f"diff_corrected_{exp}.nii"
    bval_file = f"diff_corrected_{exp}.bval"
    bvec_file = f"diff_corrected_{exp}.bvec"
    mprage_file = f"MPRAGE.nii"
    return diffusion_file, bval_file, bvec_file, mprage_file


def diffusion_correct_siemens_eddy(subj_folder, diffusion_file, bval_file, bvec_file):
    acqr_file = os_path_2_fsl(r"F:\Hila\TDI\siemens\eddy_files\datain.txt")
    index_file = os_path_2_fsl(r"F:\Hila\TDI\siemens\eddy_files\index.txt")
    subj_folder_fsl = os_path_2_fsl(subj_folder)
    # Create b0 file:
    create_b0_file(subj_folder_fsl, diffusion_file)
    # Topup:
    if not os.path.exists(os.path.join(subj_folder, "topup_b0_iout.nii")):
        topup(subj_folder_fsl, acqr_file)
    # BET:
    diffusion_bet(subj_folder_fsl)
    # Eddy:
    if not os.path.exists(os.path.join(subj_folder, "diff_corrected.nii")):
        eddy_correct(
            subj_folder_fsl, diffusion_file, acqr_file, index_file, bval_file, bvec_file
        )

    for file in os.listdir(subj_folder):
        if "eddy" in file or "topup" in file:
            os.remove(os.path.join(subj_folder, file))


def diffusion_correct_siemens(subj_folder, diffusion_file, bval_file, bvec_file):
    # motion correction:
    data, affine = load_nifti(os.path.join(subj_folder, diffusion_file))
    bvals, bvecs = read_bvals_bvecs(
        os.path.join(subj_folder, bval_file), os.path.join(subj_folder, bvec_file)
    )
    motion_correction_fname = os.path.join(subj_folder, "motion_correction.nii.gz")

    if os.path.exists(motion_correction_fname):
        data_corrected, reg_affines = load_nifti(motion_correction_fname)
    else:
        gtab = gradient_table(bvals, bvecs)
        data_corrected, reg_affines = motion_correction(data, gtab, affine)
        save_nifti(
            motion_correction_fname,
            data_corrected.get_fdata(),
            data_corrected.affine,
        )
        reg_affines = data_corrected.affine
        data_corrected = data_corrected.get_fdata()
    # Denoising:
    mask = data_corrected[..., 0] > 100
    sigma = estimate_sigma(data_corrected, N=32)
    den = nlmeans(
        data_corrected,
        sigma=sigma,
        mask=mask,
        patch_radius=1,
        block_radius=2,
        rician=True,
    )
    save_nifti(
        os.path.join(subj_folder, "motion_correction_denoised.nii.gz"),
        den,
        reg_affines,
    )


def create_b0_file(subj_folder, diffusion_file):
    cmd = rf'bash -lc "fslroi {subj_folder}/{diffusion_file} {subj_folder}/b0_1 0 1"'
    os.system(cmd)
    cmd = rf'bash -lc "fslroi {subj_folder}/{diffusion_file} {subj_folder}/b0_2 82 1"'
    os.system(cmd)
    cmd = rf'bash -lc "fslmerge -t {subj_folder}/All_b0.nii {subj_folder}//b0_1.nii {subj_folder}//b0_2.nii"'
    os.system(cmd)


def topup(subj_folder, acqr_file):
    config_file = os_path_2_fsl(r"F:\Hila\TDI\siemens\eddy_files\config.txt")
    cmd = rf'bash -lc "topup --imain={subj_folder}/All_b0.nii --datain={acqr_file} --config={config_file} --out={subj_folder}/topup_b0 --iout={subj_folder}/topup_b0_iout --fout={subj_folder}/topup_b0_fout"'
    os.system(cmd)


def diffusion_bet(subj_folder):
    cmd = rf'bash -lc "fslmaths {subj_folder}/topup_b0_iout -Tmean {subj_folder}/mean_diff"'
    os.system(cmd)
    cmd = rf'bash -lc "bet {subj_folder}/mean_diff {subj_folder}/mean_diff_brain -m -f 0.5 -g 0.05"'
    os.system(cmd)


def eddy_correct(
    subj_folder, diffusion_file, acqr_file, index_file, bval_file, bvec_file
):
    cmd = rf'bash -lc "eddy_openmp --imain={subj_folder}/{diffusion_file} --mask={subj_folder}/mean_diff_brain_mask --index={index_file} --acqp={acqr_file} --bvecs={subj_folder}/{bvec_file} --bvals={subj_folder}/{bval_file} --fwhm=0 --topup={subj_folder}/topup_b0 --flm=quadratic --out={subj_folder}/diff_corrected.nii --data_is_shelled"'
    os.system(cmd)


def bet_4_regis_mprage(subj_folder, mprage_file, diffusion_file):
    subj_mprage = os_path_2_fsl(os.path.join(subj_folder, mprage_file))
    subj_diffusion = os_path_2_fsl(os.path.join(subj_folder, diffusion_file))
    in_brain = subj_mprage[:-4]
    out_brain = subj_mprage[:-4] + "_brain"
    diff_file_1st = subj_diffusion[:-7] + "_1st"

    # BET for registered MPRAGE:
    cmd = 'bash -lc "bet {0} {1} {2} {3}"'.format(
        in_brain, out_brain, "-f 0.3", "-g -0.4"
    )
    cmd = cmd.replace(os.sep, "/")
    os.system(cmd)

    # save first corrected diff:
    cmd = rf'bash -lc "fslroi {subj_diffusion} {diff_file_1st} 0 1"'
    os.system(cmd)

    return subj_mprage, out_brain, diff_file_1st


def bet_4_corrected_diff(diff_file_1st):
    out_brain = diff_file_1st + "_brain"
    cmd = 'bash -lc "bet {0} {1} {2}"'.format(diff_file_1st, out_brain, "-f 0.3")
    cmd = cmd.replace(os.sep, "/")
    os.system(cmd)
    return out_brain


def reg_mprage_2_diff(subj_folder, subj_mprage, diffusion_1st, out_brain):
    """Registration from MPRAGE to 1st CHARMED scan using inverse matrix of CHARMED to MPRAGE registration:
    From CHARMED to MPRAGE:"""
    out_registered, out_registered_mat = reg_from_chm_2_mprage(
        subj_folder, out_brain, diffusion_1st
    )
    """Creation of inverse matrix:  """
    inv_mat = create_inv_mat(out_registered_mat)
    """From MPRAGE to CHARMED using the inverse matrix: """
    mprage_registered = reg_from_mprage_2_chm_inv(
        subj_mprage, out_brain, diffusion_1st, inv_mat
    )

    return mprage_registered


def reg_from_chm_2_mprage(
    subj_folder,
    mprage_file,
    diffusion_1st,
    out_registered=None,
    out_registered_mat=None,
):
    if out_registered == None:
        out_registered = os_path_2_fsl(subj_folder + "/rdiff_corrected_1st.nii")
    if out_registered_mat == None:
        out_registered_mat = out_registered[:-4] + ".mat"
    options = "-bins 256 -cost normmi -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12"

    cmd = f'bash -lc "flirt -ref {mprage_file} -in {diffusion_1st} -out {out_registered} -omat {out_registered_mat} {options}"'
    cmd = cmd.replace(os.sep, "/")
    os.system(cmd)
    return out_registered, out_registered_mat


def reg_from_mprage_2_chm_inv(subj_mprage, out_brain, diffusion_1st, inv_mat):
    mprage_registered = subj_mprage[:-4] + "_brain_reg.nii"
    cmd = f'bash -lc "flirt -in {out_brain} -ref {diffusion_1st} -out {mprage_registered} -applyxfm -init {inv_mat}"'
    cmd = cmd.replace(os.sep, "/")
    os.system(cmd)

    return mprage_registered


def atlas_registration(subj_folder, mprage_registered, atlas_type):
    subj_folder_fsl = os_path_2_fsl(subj_folder)
    atlas_label, atlas_template = atlas_files(atlas_type)
    atlas_label = os_path_2_fsl(atlas_label)
    atlas_template = os_path_2_fsl(atlas_template)
    if os.path.exists(os.path.join(subj_folder, "r" + atlas_label.split(sep="/")[-1])):
        return
    """Registration from atlas to regisered MPRAGE:
        flirt for atlas to registered MPRAGE for primary guess:  """
    (
        atlas_brain,
        atlas_registered_flirt,
        atlas_registered_flirt_mat,
    ) = flirt_primary_guess(subj_folder_fsl, atlas_template, mprage_registered)
    """fnirt for atlas based on flirt results:    """
    warp_name = fnirt_from_atlas_2_subj(
        subj_folder_fsl + "/",
        mprage_registered,
        atlas_brain,
        atlas_registered_flirt_mat,
        cortex_only=False,
    )

    """apply fnirt warp on atlas labels:   """
    atlas_labels_registered = apply_fnirt_warp_on_label(
        subj_folder_fsl, atlas_label, mprage_registered, warp_name
    )


def atlas_files(atlas_type):
    if atlas_type == "yeo7_200":
        atlas_label = r"G:\data\atlases\yeo\yeo7_200\yeo7_200_atlas.nii"
        atlas_template = r"G:\data\atlases\yeo\yeo7_200\Schaefer_template.nii"

    elif atlas_type == "yeo7_1000":
        atlas_label = r"G:\data\atlases\yeo\yeo7_1000\yeo7_1000_atlas.nii"
        atlas_template = r"G:\data\atlases\yeo\yeo7_1000\Schaefer_template.nii"

    elif atlas_type == "yeo17_1000":
        atlas_label = r"G:\data\atlases\yeo\yeo17_1000\yeo17_1000_atlas.nii"
        atlas_template = r"G:\data\atlases\yeo\yeo17_1000\Schaefer_template.nii"

    elif atlas_type == "bna":
        atlas_label = r"G:\data\atlases\BNA\BN_Atlas_274_combined_1mm.nii"
        atlas_template = r"G:\data\atlases\BNA\MNI152_T1_1mm.nii"

    elif atlas_type == "bnacor":
        atlas_label = r"G:\data\atlases\BNA\newBNA_Labels.nii"
        atlas_template = r"G:\data\atlases\BNA\MNI152_T1_1mm.nii"

    elif atlas_type == "yeo7_100":
        atlas_label = r"G:\data\atlases\yeo\yeo7_100\yeo7_100_atlas.nii"
        atlas_template = r"G:\data\atlases\yeo\yeo7_100\Schaefer_template.nii"

    atlas_label = os_path_2_fsl(atlas_label)
    atlas_template = os_path_2_fsl(atlas_template)

    return atlas_label, atlas_template
