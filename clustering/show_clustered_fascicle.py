from os.path import join as pjoin
from FT.weighted_tracts import *
from FT.clustering.cluster_fascicles import compute_clustering_model, clustering_input
from FT.single_fascicle_vizualization import show_fascicles_wholebrain

class ClusterBundle:

    def __init__(self,folder_name,n, cluster_method, dist_method, bundle_name, weight_by = '1.5_2_AxPasi5'):
        self.folder_name = folder_name
        self.bundle_folder = folder_name+'\streamlines'
        self.bundle_name = bundle_name
        self.subj = n
        self.cluster_method = cluster_method
        self.dist_method = dist_method
        self.weight_by = weight_by

    def load_bundle(self):
        file_list = os.listdir(self.bundle_folder)
        for file in file_list:
            if self.bundle_name in file and file.endswith('.trk'):
                self.bundle_file = pjoin(self.bundle_folder, file)
                break

        self.bundle = nib.load(self.bundle_file).get_data()
        self.affine = nib.load(self.bundle_file).affine


    def load_vols_from_ori(self):
        _, _, _, _, _, self.nii_file, self.bvec_file = load_dwi_files(self.folder_name)
        self.ori_vols = weighting_streamlines(self.folder_name, self.bundle, self.bvec_file)

    def cluster_model_input(self):
        self.tract_num = len(self.bundle._lengths)
        self.model_input = clustering_input(self.dist_method,self.tract_num,self.bundle,self.ori_vols)

    def cluster_bundle(self,n):
        self.n_clusters = n
        self.model = compute_clustering_model(self.cluster_method,self.model_input,n)


    def show_clustered(self):
        show_fascicles_wholebrain(self.bundle, self.model.labels_, self.folder_name, self.bundle_name+'_kmeans_all'+str(self.n_clusters), downsamp=1, scale=[0,self.n_clusters-1],hue=[0.15,1])

    def show_clusteres_mean_caliber(self):
        self.labels_vec = self.model.labels_
        self.labels = list(set(self.labels_vec))
        vec_mean_vol = np.zeros(len(self.ori_vols))
        v_list = []
        for l in self.labels:
            where_label = self.labels_vec == l
            l_idx = [i for i, x in enumerate(where_label) if x]
            vols = [v for v, x in zip(self.ori_vols, where_label) if x]
            mean_vol = np.mean(vols)
            std_vol = np.std(vols)
            v_list.append(vols)
            print(f'mean:{mean_vol}, std:{std_vol}')
            for i in l_idx:
                vec_mean_vol[i] = mean_vol
        self.ori_mean_vols = vec_mean_vol

        #show_fascicles_wholebrain(streamlines, vec_mean_vol, folder_name, file_name, downsamp=1, scale=[5, 7],
         #                         hue=[0.25, -0.05], saturation=[0.1, 1])


