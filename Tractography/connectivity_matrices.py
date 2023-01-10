import numpy as np
import os


class ConMatNodes():

    def __init__(self,atlas,index_to_text_file=None):
        self.atlas = atlas

        if not index_to_text_file:
            if atlas == 'bna':
                self.index_to_text_file = r'G:\data\atlases\BNA\BNA_with_cerebellum.csv'
                self.nodes_labels_bna()
                self.lab = r'\rBN_Atlas_274_combined_1mm.nii'

            elif atlas == 'bnacor':
                self.index_to_text_file = r'G:\data\atlases\BNA\BNA_with_cerebellum.csv'
                self.nodes_labels_bnacor()
                self.lab = r'\rnewBNA_Labels.nii'

            elif atlas == 'mega':
                self.index_to_text_file = r'G:\data\atlases\megaatlas\megaatlascortex2nii.txt'
                self.nodes_labels_mega()
                self.lab = r'\rMegaAtlas_cortex_Labels.nii'

            elif atlas == 'aal3':
                self.index_to_text_file = r'G:\data\atlases\aal3\AAL3\AAL3v1_1mm.nii.txt'
                self.nodes_labels_aal3()
                self.lab = r'\rAAL3_highres_atlas.nii'

            elif atlas == 'yeo7_200':
                self.index_to_text_file = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
                self.nodes_labels_yeo()
                self.lab = r'\ryeo7_200_atlas.nii'

            elif atlas == 'yeo7_1000':
                self.index_to_text_file = r'G:\data\atlases\yeo\yeo7_1000\index2label.txt'
                self.nodes_labels_yeo()
                self.lab = r'\ryeo7_1000_atlas.nii'

            elif atlas == 'yeo17_1000':
                self.index_to_text_file = r'G:\data\atlases\yeo\yeo17_1000\index2label.txt'
                self.nodes_labels_yeo()
                self.lab = r'\ryeo17_1000_atlas.nii'

    def nodes_labels_aal3(self):
        labels_file = open(self.index_to_text_file, 'r', errors='ignore')
        labels_name = labels_file.readlines()
        labels_file.close()
        labels_table = []
        labels_headers = []
        idx = []
        for line in labels_name:
            if not line[0] == '#':
                labels_table.append([col for col in line.split() if col])

        for l in labels_table:
            if len(l) == 3:
                head = l[1]
                labels_headers.append(head)
                idx.append(int(l[0]) - 1)
        # pop over not assigned indices (in aal3):
        self.labels_headers = labels_headers
        idx = np.asarray(idx)
        first = idx > 35
        second = idx > 81
        idx[first] -= 2
        idx[second] -= 2
        self.idx = list(idx)

    def nodes_labels_yeo(self):
        labels_file = open(self.index_to_text_file, 'r', errors='ignore')
        labels_name = labels_file.readlines()
        labels_file.close()
        labels_table = []
        labels_headers = []
        idx = []
        for line in labels_name:
            if not line[0] == '#':
                labels_table.append([col for col in line.split() if col])

        for l in labels_table:
            if len(l) >= 3:
                head = l[1]
                labels_headers.append(head)
                idx.append(int(l[0]) - 1)
        self.labels_headers = labels_headers
        self.idx = list(idx)

    def nodes_labels_bna(self):
        labels_file = open(self.index_to_text_file, 'r', errors='ignore')
        labels_name = labels_file.readlines()
        labels_file.close()
        labels_table = []
        labels_headers = []
        idx = []

        for line in labels_name:
            if not line[0] == '#':
                labels_table.append([col for col in line.split() if col])

        labels_table = labels_table[1:247:2] + labels_table[247::] + labels_table[2:247:2]

        for l in labels_table:
            lparts = l[0].split(',')
            idx.append(int(lparts[0]))
            labels_headers.append(lparts[1])
        self.idx = list(idx)
        self.labels_headers = labels_headers

    def nodes_labels_bnacor(self):
        labels_file = open(self.index_to_text_file, 'r', errors='ignore')
        labels_name = labels_file.readlines()
        labels_file.close()
        labels_table = []
        labels_headers = []
        idx = []

        for line in labels_name:
            if not line[0] == '#':
                labels_table.append([col for col in line.split() if col])

        labels_table = labels_table[1:211:2] + labels_table[2:211:2]

        for l in labels_table:
            lparts = l[0].split(',')
            idx.append(int(lparts[0]))
            labels_headers.append(lparts[1])
        self.idx = list(idx)
        self.labels_headers = labels_headers

    def nodes_labels_mega(self):
        labels_file = open(self.index_to_text_file, 'r', errors='ignore')
        labels_name = labels_file.readlines()
        labels_file.close()
        labels_table = []
        labels_headers = []
        idx = []
        for line in labels_name:
            if not line[0] == '#':
                labels_table.append([col for col in line.split("\t") if col])
            elif 'ColHeaders' in line:
                labels_headers = [col for col in line.split(" ") if col]
                labels_headers = labels_headers[2:]
        for l in labels_table:
            head = l[1]
            labels_headers.append(head[:-1])
            idx.append(int(l[0]) - 1)
        self.idx = idx
        self.labels_headers = labels_headers

    def nodes_by_idx(self,folder_name):
        import nibabel as nib
        lab = folder_name+self.lab
        lab_file = nib.load(lab)
        lab_labels = lab_file.get_fdata()
        self.affine = lab_file.affine
        lab_labels_index = [labels for labels in lab_labels]
        self.lab_labels_index = np.asarray(lab_labels_index, dtype='int')
        return self.affine, self.lab_labels_index


class ConMat():

    def __init__(self,atlas,subj_folder,diff_file = 'data.nii',cm_name = None, index_to_text_file=None, tract_name = 'HCP_tracts.tck',streamlines=None):
        from dipy.tracking import utils
        from Tractography.files_loading import load_ft

        self.subj_folder = subj_folder
        self.atlas = atlas
        cm_nodes = ConMatNodes(self.atlas, index_to_text_file)
        self.index_to_text_file = cm_nodes.index_to_text_file
        self.labels = cm_nodes.labels_headers
        self.idx = cm_nodes.idx
        self.affine, self.lab_labels_index = cm_nodes.nodes_by_idx(self.subj_folder)
        nii_ref = os.path.join(subj_folder,diff_file)
        if not cm_name:
            if not streamlines:
                tract_path = os.path.join(self.subj_folder, 'streamlines', tract_name)
                streamlines = load_ft(tract_path, nii_ref)

            m, self.grouping = utils.connectivity_matrix(streamlines, self.affine, self.lab_labels_index,
                                                         return_mapping=True,
                                                         mapping_as_streamlines=True)
            self.fix_cm(m)
            self.ord_cm()
            self.norm_cm = 1 / self.ord_cm

        else:
            cm_file_name = f'{subj_folder}cm{os.sep}{cm_name}.npy'
            self.cm = np.load(cm_file_name)

    def fix_cm(self,m):
        mm = m[1:]
        mm = mm[:, 1:]
        if 'aal3' in self.atlas:
            mm = np.delete(mm, [34, 35, 80, 81], 0)
            mm = np.delete(mm, [34, 35, 80, 81], 1)
        self.cm = mm

    def ord_cm(self):
        self.ord_cm = self.cm[self.idx]
        self.ord_cm = self.ord_cm[:,self.idx]

    def save_cm(self,fig_name,mat_type='cm_ord'):
        self.cm_dir = f'{self.subj_folder}{os.sep}cm{os.sep}'
        if not os.path.exists(self.cm_dir):
            os.mkdir(self.cm_dir)
        self.fig_name = self.cm_dir+fig_name+'_'+mat_type
        if mat_type == 'cm':
            np.save(self.fig_name, self.cm)
        elif mat_type == 'cm_ord':
            np.save(self.fig_name, self.ord_cm)
            self.save_lookup()
        elif mat_type == 'cm_norm':
            np.save(self.fig_name, self.norm_cm)
            self.save_lookup()
        else:
            print("Couldn't recognize which mat to save.\n Please specify one of the following: \n 'cm' : for original matrix\n 'cm_ord' : for matrix ordered using atlas indices (default)\n 'cm_norm' : for normalized matrix")

    def save_lookup(self):
        np.save(self.cm_dir+self.atlas+'_cm_ord_lookup', self.idx)

    def draw_con_mat(self,mat_type, show=True, cmap_colors = 'YlOrRd'):
        import matplotlib.colors as colors
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        if mat_type == 'cm':
            data = self.cm
        elif mat_type == 'cm_ord':
            data = self.ord_cm
        elif mat_type == 'cm_norm':
            data = self.norm_cm

        max_val = np.max(data[np.isfinite(data)])

        data[~np.isfinite(data)] = max_val
        mat_title = 'Number of tracts weighted connectivity matrix'
        plt.figure(1, [30, 24])
        cmap = cm.get_cmap(cmap_colors).copy()
        cmap.set_over('black')
        plt.imshow(data, interpolation='nearest', cmap=cmap, origin='upper', norm = colors.LogNorm(vmax=0.99*max_val))
        plt.colorbar()
        plt.xticks(ticks=np.arange(0, len(data), 1), labels=self.labels)
        plt.yticks(ticks=np.arange(0, len(data), 1), labels=self.labels)
        plt.title(mat_title, fontsize=32)
        plt.tick_params(axis='x', pad=10.0, labelrotation=90, labelsize=11)
        plt.tick_params(axis='y', pad=10.0, labelsize=11)
        try:
            plt.savefig(self.fig_name+'.png')
        except AttributeError:
            plt.savefig(f'{self.subj_folder}cm{os.sep}{mat_type}.png')
        if show:
            plt.show()
        else:
            plt.clf()


class WeightConMat(ConMat):

    def __init__(self,weight_by, atlas,subj_folder,diff_file = 'data.nii',index_to_text_file=None, norm_factor = 8.75, tract_name = 'HCP_tracts.tck', streamlines=None):

        self.weight_by = weight_by
        self.factor = norm_factor
        if not index_to_text_file:
            super().__init__(atlas, subj_folder, diff_file, tract_name = tract_name, streamlines=streamlines)
        else:
            super().__init__(atlas, subj_folder,diff_file, index_to_text_file=index_to_text_file, tract_name = tract_name, streamlines=streamlines)

        if self.weight_by == 'dist':
            self.dist_cm()
        else:
            self.weight_cm()

    def weight_cm(self):
        from Tractography.files_loading import load_weight_by_img
        from dipy.tracking.streamline import values_from_volume

        weight_by_data, affine = load_weight_by_img(self.subj_folder, self.weight_by)
        m_weighted = np.zeros((len(self.idx), len(self.idx)), dtype='float64')
        for pair, tracts in self.grouping.items():
            if pair[0] == 0 or pair[1] == 0:
                continue
            else:
                mean_vol_per_tract = []
                vol_per_tract = values_from_volume(weight_by_data, tracts, affine=affine)
                for s in vol_per_tract:
                    s = np.asanyarray(s)
                    mean_vol_per_tract.append(np.nanmean(s))
                mean_path_vol = np.nanmean(mean_vol_per_tract)
                if 'aal3' in self.atlas:
                    r = pair[0] - 1
                    c = pair[1] - 1

                    if r > 81:
                        r -= 4
                    elif r > 35:
                        r -= 2

                    if c > 81:
                        c -= 4
                    elif c > 35:
                        c -= 2

                    m_weighted[r, c] = mean_path_vol
                    m_weighted[c, r] = mean_path_vol

                else:
                    m_weighted[pair[0] - 1, pair[1] - 1] = mean_path_vol
                    m_weighted[pair[1] - 1, pair[0] - 1] = mean_path_vol

        self.cm = m_weighted
        super().ord_cm()
        self.norm_cm = 1 / (self.ord_cm * self.factor)  # 8.75 - axon diameter 2 ACV constant

    def dist_cm(self):
        from dipy.tracking.utils import length

        m_weighted = np.zeros((len(self.idx), len(self.idx)), dtype='float64')
        for pair, tracts in self.grouping.items():
            if pair[0] == 0 or pair[1] == 0:
                continue
            else:
                s_len = 1/np.nanmean([s.shape[0] for s in tracts])
                if not np.isfinite(s_len):
                    s_len = 0

            if 'aal3' in self.atlas:
                r = pair[0] - 1
                c = pair[1] - 1

                if r > 81:
                    r -= 4
                elif r > 35:
                    r -= 2

                if c > 81:
                    c -= 4
                elif c > 35:
                    c -= 2

                m_weighted[r, c] = s_len
                m_weighted[c, r] = s_len

            else:
                m_weighted[pair[0] - 1, pair[1] - 1] = s_len
                m_weighted[pair[1] - 1, pair[0] - 1] = s_len

            self.cm = m_weighted
            super().ord_cm()


    def draw_con_mat(self,mat_type, show=True, cmap_colors = 'YlOrRd'):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if mat_type == 'cm':
            data = self.cm
        elif mat_type == 'cm_ord':
            data = self.ord_cm
        elif mat_type == 'cm_norm':
            data = self.norm_cm

        max_val = np.max(data[np.isfinite(data)])

        data[~np.isfinite(data)] = max_val

        mat_title = f'Weighted connectivity matrix by {self.weight_by}'
        plt.figure(1, [30, 24])
        cmap = cm.get_cmap(cmap_colors).copy()
        cmap.set_over('black')
        plt.imshow(data, interpolation='nearest', cmap=cmap, origin='upper',vmax=0.99*max_val)
        plt.colorbar()
        plt.xticks(ticks=np.arange(0, len(data), 1), labels=self.labels)
        plt.yticks(ticks=np.arange(0, len(data), 1), labels=self.labels)
        plt.title(mat_title, fontsize=32)
        plt.tick_params(axis='x', pad=10.0, labelrotation=90, labelsize=11)
        plt.tick_params(axis='y', pad=10.0, labelsize=11)
        plt.savefig(self.fig_name+'.png')
        if show:
            plt.show()










