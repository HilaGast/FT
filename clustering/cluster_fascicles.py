
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from dipy.tracking.streamline import set_number_of_points
from FT.weighted_tracts import *
from FT.single_fascicle_vizualization import streamline_mean_fascicle_value_weighted, show_fascicles_wholebrain
from FT.clustering.poly_representaion_fibers import poly_xyz_vec_calc
from scipy.stats import f_oneway
from dipy.segment.bundles import bundles_distances_mam


def clustering_input(method,tracts_num,streamlines,vec_vols):
    if method == 'cartesian':
        subsample = 100
        subsamp_sls = set_number_of_points(streamlines, subsample)

        f_mat = np.zeros([tracts_num, subsample * 3 * 2])
        # f_mat = np.zeros([tracts_num,subsample*3])

        for i in range(tracts_num):
            locations = subsamp_sls[i].flatten()
            diameter = np.ones(subsample*3)*vec_vols[i]
            f_vec = np.append(locations, diameter)
            #f_vec = locations
            f_mat[i] = f_vec

    elif method == 'polynomial':
        degree = 3
        f_mat = np.zeros([tracts_num, (degree + 1) * 3 + 1])
        # f_mat = np.zeros([tracts_num,(degree+1)*3])
        # f_mat = np.zeros([tracts_num,1])

        for i in range(tracts_num):
            locations = poly_xyz_vec_calc(streamlines[i], degree)
            diameter = vec_vols[i] * (3 * (degree + 1)) **2
            f_vec = np.append(locations, diameter)
            # f_vec = locations
            # f_vec = diameter

            f_mat[i] = f_vec

    elif method == 'mam':

        f_mat = np.zeros([tracts_num,4])
        #f_mat = np.zeros([tracts_num,3])

        m = bundles_distances_mam(streamlines, streamlines, 'avg')
        sum_of_m = np.sum(m, axis=0)
        sum_of_m = list(sum_of_m)

        imax = sum_of_m.index(max(sum_of_m))
        imin = sum_of_m.index(min(sum_of_m))
        farest = m[:, imax] + m[:, imin]
        farest = list(farest)
        ifar = farest.index(max(farest))

        locs = np.asarray([m[:, imin], m[:, imax], m[:, ifar]]).T

        #iclose1 = farest.index(min(farest))
        #cl2 = list(m[:, ifar] + m[:, imin])
        #iclose2 = cl2.index(min(cl2))
        # locs = np.asarray([m[:,imin],m[:,imax],m[:,ifar],m[:,iclose1],m[:,iclose2]]).T

        f_mat[:,:3] = locs
        f_mat[:,3] = np.asarray(vec_vols)**3

        #f_mat = locs
        #f_mat = vec_vols
        return f_mat


def show_23456_groups(method,streamlines,folder_name,X,fascicle):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        show_fascicles_wholebrain(streamlines, kmeans.labels_, folder_name, fascicle+'_kmeans_all2', downsamp=1, scale=[0,1],hue=[0.15,1])

        kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
        show_fascicles_wholebrain(streamlines, kmeans.labels_, folder_name,  fascicle+'_kmeans_all3', downsamp=1, scale=[0,2],hue=[0.15,1])

        kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
        show_fascicles_wholebrain(streamlines, kmeans.labels_, folder_name,  fascicle+'_kmeans_all4', downsamp=1, scale=[0,3],hue=[0.15,1])

        kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
        show_fascicles_wholebrain(streamlines, kmeans.labels_, folder_name,  fascicle+'_kmeans_all5', downsamp=1, scale=[0,4],hue=[0.15,1])

        kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
        show_fascicles_wholebrain(streamlines, kmeans.labels_, folder_name,  fascicle+'_kmeans_all6', downsamp=1, scale=[0,5],hue=[0.15,1])

    elif method == 'agglomerative':

        cluster = AgglomerativeClustering(n_clusters=6, distance_threshold=None, compute_full_tree='auto', affinity='euclidean', linkage='ward').fit(X)
        show_fascicles_wholebrain(streamlines, cluster.labels_, folder_name, fascicle+'_agg_all6', downsamp=1, scale=[0,5],hue=[0.15,1])

        cluster = AgglomerativeClustering(n_clusters=5, distance_threshold=None, compute_full_tree='auto', affinity='euclidean', linkage='ward').fit(X)
        show_fascicles_wholebrain(streamlines, cluster.labels_, folder_name, fascicle+'_agg_all5', downsamp=1, scale=[0,4],hue=[0.15,1])

        cluster = AgglomerativeClustering(n_clusters=4, distance_threshold=None, compute_full_tree='auto', affinity='euclidean', linkage='ward').fit(X)
        show_fascicles_wholebrain(streamlines, cluster.labels_, folder_name, fascicle+'_agg_all4', downsamp=1, scale=[0,3],hue=[0.15,1])

        cluster = AgglomerativeClustering(n_clusters=3, distance_threshold=None, compute_full_tree='auto', affinity='euclidean', linkage='ward').fit(X)
        show_fascicles_wholebrain(streamlines, cluster.labels_, folder_name, fascicle+'_agg_all3', downsamp=1, scale=[0,2],hue=[0.15,1])

        cluster = AgglomerativeClustering(n_clusters=2, distance_threshold=None, compute_full_tree='auto', affinity='euclidean', linkage='ward').fit(X)
        show_fascicles_wholebrain(streamlines, cluster.labels_, folder_name, fascicle+'_agg_all2', downsamp=1, scale=[0,1],hue=[0.15,1])


def compute_clustering_model(method,X,n):
    if method == 'kmeans':
        model = KMeans(n_clusters=n, random_state=0).fit(X)

    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n, distance_threshold=None, compute_full_tree='auto', affinity='euclidean', linkage='ward').fit(X)

    return model


def weighted_clusters(model,streamlines,vec_vols, file_name = ''):
    labels = list(set(model.labels_))
    labels_vec = model.labels_
    vec_mean_vol = np.zeros(len(vec_vols))
    v_list = []
    for l in labels:
        where_label = labels_vec == l
        l_idx = [i for i,x in enumerate(where_label) if x]
        vols = [v for v, x in zip(vec_vols, where_label) if x]
        mean_vol = np.mean(vols)
        std_vol = np.std(vols)
        v_list.append(vols)
        print(f'mean:{mean_vol}, std:{std_vol}')
        for i in l_idx:
            vec_mean_vol[i] = mean_vol
    #stats,p=f_oneway(v_list[0],v_list[1],v_list[2])
    #print(f'F={stats:.3f}, p={p:.5f}')
    show_fascicles_wholebrain(streamlines, vec_mean_vol, folder_name, file_name, downsamp=1, scale=[5,6],hue = [0.25,-0.05],saturation = [0.1, 1])

if __name__ == '__main__':
    main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep'
    folder_name = main_folder + all_subj_folders[0]
    n = all_subj_names[0]
    nii_file = load_dwi_files(folder_name)[5]
    fascicle = 'cing'
    streamlines,vec_vols = streamline_mean_fascicle_value_weighted(folder_name, n, nii_file, fascicle)
    method = 'mam'
    tracts_num = streamlines.__len__()

    X = clustering_input(method,tracts_num,streamlines,vec_vols)
    methods = ['agglomerative','kmeans']
    method = methods[1]

    show_23456_groups(method,streamlines,folder_name,X,fascicle)

    g=[2,3,4,5,6]
    for n in g:
        model = compute_clustering_model(method,X,n)
        weighted_clusters(model,streamlines,vec_vols, file_name = 'clustered_'+fascicle+'_'+str(n))

