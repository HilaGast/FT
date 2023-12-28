import os, glob
import numpy as np
from HCP_network_analysis.HCP_cm.euclidean_distance_matrix import (
    find_labels_file,
    find_labels_centroids,
    euc_dist_mat,
)
from Tractography.connectivity_matrices import ConMat, WeightConMat
from Tractography.fiber_tracking import fiber_tracking_parameters, Tractography
from Tractography.files_loading import load_ft
from Tractography.tractography_vis import show_average_tract_by_atlas_cm
from fsl.file_prep import fast_seg, os_path_2_fsl
from fsl.file_prep_siemens import (
    diffusion_correct_siemens,
    subj_files,
    bet_4_regis_mprage,
    reg_mprage_2_diff,
    atlas_registration,
)


def preprocessing(subj_folder, exp, atlas_types):
    diffusion_file, bval_file, bvec_file, mprage_file = subj_files(exp)

    if not os.path.exists(os.path.join(subj_folder, "diff_corrected.nii")):
        diffusion_correct_siemens(subj_folder, diffusion_file, bval_file, bvec_file)

    if not os.path.exists(os.path.join(subj_folder, "MPRAGE_brain_reg.nii")):
        subj_mprage, out_brain, diffusion_1st = bet_4_regis_mprage(
            subj_folder, mprage_file, "diff_corrected.nii"
        )
        mprage_registered = reg_mprage_2_diff(
            subj_folder, subj_mprage, diffusion_1st, out_brain
        )
    else:
        mprage_registered = os_path_2_fsl(
            os.path.join(subj_folder, "MPRAGE_brain_reg.nii")
        )

    for atlas_type in atlas_types:
        atlas_registration(subj_folder, mprage_registered, atlas_type)

    if not os.path.exists(os.path.join(subj_folder, f"MPRAGE_brain_reg_seg.nii")):
        fast_seg(mprage_registered)


def fiber_tracking(subj_folder):
    diff_data = os.path.join(subj_folder, "diff_corrected.nii")
    if not os.path.exists(subj_folder + f"{os.sep}streamlines{os.sep}wb_csd_fa.tck"):
        tissue_labels_file_name = "MPRAGE_brain_reg_seg.nii"
        pve_file_name = "MPRAGE_brain_reg_pve"
        parameters_dict = fiber_tracking_parameters(
            max_angle=30,
            sh_order=8,
            seed_density=4,
            streamlines_lengths_mm=[50, 500],
            step_size=1,
            sh_order=6,
            fa_th=0.18,
        )
        tracts = Tractography(
            subj_folder,
            "fa",
            "csd",
            "wb",
            parameters_dict,
            diff_data,
            tissue_labels_file_name=tissue_labels_file_name,
            pve_file_name=pve_file_name,
        )
        tracts.fiber_tracking()
        streamlines = tracts.streamlines
    else:
        tract_name = os.path.join(subj_folder, "streamlines", "wb_csd_fa.tck")
        streamlines = load_ft(tract_name, diff_data)

    return streamlines


def weighted_con_mat(subj_folder, atlas_type, streamlines):
    diffusion_data = os.path.join(subj_folder, "diff_corrected.nii")
    for atlas in atlas_type:
        if not os.path.exists(f"{subj_folder}{os.sep}cm{os.sep}num_{atlas}_cm_ord.npy"):
            cm = ConMat(
                atlas=atlas,
                diff_file=diffusion_data,
                subj_folder=subj_folder,
                tract_name="wb_csd_fa.tck",
                streamlines=streamlines,
            )
            cm.save_cm(fig_name=f"num_{atlas}", mat_type="cm_ord")

        if not os.path.exists(f"{subj_folder}{os.sep}cm{os.sep}add_{atlas}_cm_ord.npy"):
            cm = WeightConMat(
                weight_by="ADD",
                atlas=atlas,
                diff_file=diffusion_data,
                subj_folder=subj_folder,
                tract_name="wb_csd_fa.tck",
                streamlines=streamlines,
            )
            cm.save_cm(fig_name=f"add_{atlas}", mat_type="cm_ord")

        if not os.path.exists(
            f"{subj_folder}{os.sep}cm{os.sep}DistMode_{atlas}_cm_ord.npy"
        ):
            cm = WeightConMat(
                weight_by="distmode",
                atlas=atlas,
                diff_file=diffusion_data,
                subj_folder=subj_folder,
                tract_name="wb_csd_fa.tck",
                streamlines=streamlines,
            )
            cm.save_cm(fig_name=f"DistMode_{atlas}", mat_type="cm_ord")

        if not os.path.exists(
            f"{subj_folder}{os.sep}cm{os.sep}DistSampAvg_{atlas}_cm_ord.npy"
        ):
            cm = WeightConMat(
                weight_by="distresamp",
                atlas=atlas,
                diff_file=diffusion_data,
                subj_folder=subj_folder,
                tract_name="wb_csd_fa.tck",
                streamlines=streamlines,
            )
            cm.save_cm(fig_name=f"DistSampAvg_{atlas}", mat_type="cm_ord")

        num_cm_name = f"{subj_folder}{os.sep}cm{os.sep}num_{atlas}_cm_ord.npy"
        euc_dist_cm_name = f"{subj_folder}{os.sep}cm{os.sep}EucDist_{atlas}_cm_ord.npy"
        if not os.path.exists(euc_dist_cm_name):
            cm = np.load(num_cm_name)
            labels_file_path = find_labels_file(num_cm_name)
            label_ctd = find_labels_centroids(labels_file_path)
            euc_mat = euc_dist_mat(label_ctd, cm)
            np.save(euc_dist_cm_name, euc_mat)


def time_delay_index_cm():
    pass


if __name__ == "__main__":
    main_folder = r"F:\Hila\TDI\siemens"
    experiments = ["D31d18", "D45d13", "D60d11"]
    atlas_types = ["yeo7_100"]
    for exp in experiments:
        subj_folders = glob.glob(f"{main_folder}{os.sep}{exp}{os.sep}[C,T]*")
        # subj_folders = subj_folders[:5] + subj_folders[42:48]
        for subj_folder in subj_folders:
            print(subj_folder)
            preprocessing(subj_folder, exp, atlas_types)
            if not os.path.exists(
                subj_folder
                + f"{os.sep}cm{os.sep}DistSampAvg_{atlas_types[-1]}_cm_ord.npy"
            ):
                streamlines = fiber_tracking(subj_folder)
                weighted_con_mat(subj_folder, atlas_types, streamlines)
            # show_average_tract_by_atlas_cm(subj_folder, atlas_types, streamlines)
