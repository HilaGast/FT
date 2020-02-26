import numpy as np
import matplotlib.pyplot as plt
from FT.single_fascicle_vizualization import show_fascicles_wholebrain
from dipy.tracking.streamline import set_number_of_points
from FT.weighted_tracts import *
from FT.single_fascicle_vizualization import streamline_mean_fascicle_value_weighted


main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep'
folder_name = main_folder + all_subj_folders[0]
n = all_subj_names[0]
nii_file = load_dwi_files(folder_name)[5]

streamlines,vec_vols = streamline_mean_fascicle_value_weighted(folder_name, n, nii_file, 'ilf')

from FT.clustering.poly_representaion_fibers import *
import matplotlib.pyplot as plt
fi_all=[]
subsample = 50
subsamp_sls = set_number_of_points(streamlines, subsample)
#for i,f in enumerate(streamlines[::3]):
for i, f in enumerate(subsamp_sls[::10]):
    fi = f
    poly_xyz = poly_xyz_vec_calc(fi, degree=3)
    dist_vec = distance_vec_rep_of_fibers(fi)
    dist_mat = distance_powered_matrix(dist_vec,3)

    x = np.matmul(dist_mat,poly_xyz[0:4])
    y = np.matmul(dist_mat,poly_xyz[4:8])
    z = np.matmul(dist_mat,poly_xyz[8:])
    fi_asses = np.zeros(fi.shape)
    fi_asses[:,0] = x.T
    fi_asses[:,1] = y.T
    fi_asses[:,2] = z.T
    fi_all.append(fi_asses)

show_fascicles_wholebrain(fi_all+subsamp_sls[::10],list(np.ones(len(fi_all)))+list(np.zeros(len(subsamp_sls[::10]))),folder_name,'ilf_asses_poly',hue = [0.5,0.7],scale=[-0.2, 1])



