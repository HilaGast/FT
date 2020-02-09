from dipy.tracking.streamline import set_number_of_points
from FT.weighted_tracts import *
from FT.single_fascicle_vizualization import streamline_mean_fascicle_value_weighted, show_fascicles_wholebrain
from sklearn.cluster import KMeans


main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep'
folder_name = main_folder + all_subj_folders[0]
n = all_subj_names[0]
nii_file = load_dwi_files(folder_name)[5]

streamlines,vec_vols = streamline_mean_fascicle_value_weighted(folder_name, n, nii_file, 'cc')
subsample = 12
subsamp_sls = set_number_of_points(streamlines, subsample)
tracts_num = subsamp_sls.__len__()
f_mat = np.zeros([tracts_num,subsample*3+1])
#f_mat = np.zeros([tracts_num,subsample*3])

for i in range(tracts_num):
    locations = subsamp_sls[i].flatten()
    diameter = vec_vols[i]*36
    f_vec = np.append(locations,diameter)
    #f_vec = locations

    f_mat[i] = f_vec

#km = KMeans(n_clusters=2).fit(f_mat)
#show_fascicles_wholebrain(subsamp_sls, km.labels_, folder_name, 'slf', downsamp=1)


from sklearn.cluster import AgglomerativeClustering
from FT.single_fascicle_vizualization import show_fascicles_wholebrain
X = f_mat
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
show_fascicles_wholebrain(subsamp_sls, cluster.labels_, folder_name, 'slf', downsamp=1)



