import glob, os
from average_con_mat import calc_avg_mat
import matplotlib.pyplot as plt
import numpy as np


def average_time_mat(group_subj, group_name, main_fol, mat_type, atlas_type):
    calc_avg_mat(
        group_subj,
        mat_type,
        main_fol,
        calc_type="median",
        atlas_type=atlas_type,
        adds_for_file_name=group_name,
    )


if __name__ == "__main__":
    main_fol = "F:\Hila\TDI\siemens"
    exp = "D31d18"
    atlas_type = "yeo7_100"
    mat_type = "TDI_DistSampAvg"
    ms0_subj_fol = glob.glob(f"{main_fol}{os.sep}{exp}{os.sep}T0*")
    ms1_subj_fol = glob.glob(f"{main_fol}{os.sep}{exp}{os.sep}T1*")
    ms2_subj_fol = glob.glob(f"{main_fol}{os.sep}{exp}{os.sep}T2*")
    h_subj_fol = glob.glob(f"{main_fol}{os.sep}{exp}{os.sep}C*")
    average_time_mat(
        ms0_subj_fol,
        f"{exp}_ms0",
        main_fol + f"{os.sep}{exp}{os.sep}group_cm",
        mat_type,
        atlas_type,
    )
    average_time_mat(
        ms1_subj_fol,
        f"{exp}_ms1",
        main_fol + f"{os.sep}{exp}{os.sep}group_cm",
        mat_type,
        atlas_type,
    )
    average_time_mat(
        ms2_subj_fol,
        f"{exp}_ms2",
        main_fol + f"{os.sep}{exp}{os.sep}group_cm",
        mat_type,
        atlas_type,
    )
    average_time_mat(
        h_subj_fol,
        f"{exp}_h",
        main_fol + f"{os.sep}{exp}{os.sep}group_cm",
        mat_type,
        atlas_type,
    )

    a = np.load(
        rf"F:\Hila\TDI\siemens\{exp}\group_cm\mean_{mat_type}_{atlas_type}_{exp}_h.npy"
    )
    b = np.load(
        rf"F:\Hila\TDI\siemens\{exp}\group_cm\mean_{mat_type}_{atlas_type}_{exp}_ms0.npy"
    )

    range = (0, 100)
    plt.hist(a[a > 0], bins=50, color="blue", alpha=0.2, range=range, density=False)
    plt.hist(b[b > 0], bins=50, color="red", alpha=0.2, range=range, density=False)
    plt.hist(
        a[a > 0],
        bins=50,
        histtype="step",
        color="blue",
        linewidth=2,
        range=range,
        density=False,
    )
    plt.hist(
        b[b > 0],
        bins=50,
        histtype="step",
        color="red",
        linewidth=2,
        range=range,
        density=False,
    )
    plt.legend(["Healthy", "MS"])
    plt.show()
