from dipy.tracking.streamline import set_number_of_points
from FT.weighted_tracts import *
from FT.single_fascicle_vizualization import streamline_mean_fascicle_value_weighted, show_fascicles_wholebrain
from FT.clustering.poly_representaion_fibers import poly_xyz_vec_calc

main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep'
folder_name = main_folder + all_subj_folders[0]
n = all_subj_names[0]
nii_file = load_dwi_files(folder_name)[5]

streamlines,vec_vols = streamline_mean_fascicle_value_weighted(folder_name, n, nii_file, 'slf')

method = 'polynomial'
tracts_num = streamlines.__len__()
if method == 'cartesian':
    subsample = 50
    subsamp_sls = set_number_of_points(streamlines, subsample)

    f_mat = np.zeros([tracts_num,subsample*3+1])
    #f_mat = np.zeros([tracts_num,subsample*3])

    for i in range(tracts_num):
        locations = subsamp_sls[i].flatten()
        diameter = vec_vols[i]*36
        f_vec = np.append(locations,diameter)
        #f_vec = locations
        f_mat[i] = f_vec

elif method == 'polynomial':
    degree=3
    #f_mat = np.zeros([tracts_num,(degree+1)*3+1])
    f_mat = np.zeros([tracts_num,(degree+1)*3])
    for i in range(tracts_num):
        locations = poly_xyz_vec_calc(streamlines[i],degree)
        diameter = vec_vols[i]*(3*(degree+1))**2
        #f_vec = np.append(locations,diameter)
        f_vec = locations
        f_mat[i] = f_vec


from sklearn.cluster import AgglomerativeClustering
from FT.single_fascicle_vizualization import show_fascicles_wholebrain
X = f_mat
cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=1000, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
show_fascicles_wholebrain(streamlines, cluster.labels_, folder_name, 'slf', downsamp=1)



