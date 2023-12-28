import glob, os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    main_fol = "F:\Hila\TDI\siemens"
    exp = "D45d13"
    atlas_type = "yeo7_100"
    mat_type = "DistMode"
    ms_subj_fol = glob.glob(f"{main_fol}{os.sep}{exp}{os.sep}T*")
    h_subj_fol = glob.glob(f"{main_fol}{os.sep}{exp}{os.sep}C*")
    ms_all_subj_mat = []
    h_all_subj_mat = []
