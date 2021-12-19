from weighted_tracts import load_dwi_files
import os
import nibabel as nib


class GM_mask:

    def __init__(self,subj_fol,atlas_name='yeo7_1000',weight_by = '3_2_AxPasi7',tractography_type = 'wholebrain_5d_labmask_msmt',atlas_main_folder = r'C:\Users\Admin\my_scripts\aal\yeo'):
        self.folder_name = subj_fol #full path of subj folder
        self.tractography_type = tractography_type+'.trk'
        self.nii_file_name = load_dwi_files(self.folder_name)[5]
        self.streamline_name = self._streamline_name()
        self.weight_name = weight_by
        self.streamlines = self._load_streamlines()
        self.atlas_type = str.lower(atlas_name)
        self.atlas_main_folder = atlas_main_folder
        self.mask = self._load_gm_mask()
        if 'yeo' in self.atlas_type:
            self.atlas_file_name = os.path.join(self.folder_name, 'r' + atlas_name + '_atlas.nii')
            self.idx_to_txt = f'{self.atlas_main_folder}{os.sep}{self.atlas_type}{os.sep}index2label.txt'
            self.mni_atlas_label = f'{self.atlas_main_folder}{os.sep}{self.atlas_type}{os.sep}{self.atlas_type}_atlas.nii'
        elif 'bna' in self.atlas_type:
            self.atlas_type = atlas_name
            self.atlas_file_name = os.path.join(self.folder_name, 'rBN_Atlas_274_combined_1mm.nii')
            self.idx_to_txt = f'{self.atlas_main_folder}{os.sep}BNA_with_cerebellum.csv'
            self.mni_atlas_label = f'{self.atlas_main_folder}{os.sep}BN_Atlas_274_combined_1mm.nii'



    def endpoint_type(self,type_name):
        self.endpoint_type_name = type_name


    def _load_gm_mask(self):
        '''
        Load tissue scan (from Fast in preprocessing) to choose GM mask
        '''

        labels = load_dwi_files(self.folder_name)[3]
        mask = (labels==2)
        return mask


    def _load_streamlines(self):
        from weighted_tracts import load_ft
        '''
        Load streamlines for connectivity matrices calculations
        '''
        streamlines = load_ft(self.streamline_name, self.nii_file_name)

        return streamlines


    def _streamline_name(self):
        import glob
        tract_path = os.path.join(self.folder_name,'streamlines')
        try:
            streamline_name = glob.glob(tract_path+'*/*'+self.tractography_type)[0]
        except IndexError:
            print('No streamline file for subject')

        return streamline_name


    def _reg_atlas(self):
        '''
        If there is no registered atlas file, register atlas to subject
        '''
        from fsl.file_prep import flirt_primary_guess, fnirt_from_atlas_2_subj, apply_fnirt_warp_on_label, os_path_2_fsl,subj_files

        if os.path.exists(self.atlas_file_name):
            print(f'{self.atlas_type} is registered to subject')
        else:

            subj_folder = self.folder_name.replace(os.sep, '/')
            mprage_file_name = subj_files(subj_folder)[0]
            subj_folder = os_path_2_fsl(subj_folder+r'/')
            out_registered = subj_folder + 'r' + mprage_file_name[:-4] + '_brain.nii'

            atlas_label = self.mni_atlas_label
            if 'yeo' in self.atlas_type:
                atlas_template = f'{self.atlas_main_folder}{os.sep}{self.atlas_type}{os.sep}Schaefer_template.nii'
            elif 'bna' in self.atlas_type:
                atlas_template = f'{self.atlas_main_folder}{os.sep}MNI152_T1_1mm_brain.nii'

            atlas_label = os_path_2_fsl(atlas_label)
            atlas_template = os_path_2_fsl(atlas_template)
            atlas_brain, atlas_registered_flirt, atlas_registered_flirt_mat = flirt_primary_guess(subj_folder,
                                                                                                  atlas_template,
                                                                                                  out_registered)
            warp_name = fnirt_from_atlas_2_subj(subj_folder, out_registered, atlas_brain, atlas_registered_flirt_mat,
                                                cortex_only=False)
            atlas_labels_registered = apply_fnirt_warp_on_label(subj_folder, atlas_label, out_registered, warp_name)

            print(f'{self.atlas_type} is registered to subject')

    def _load_mni_atlas(self):
        self._reg_atlas()
        atlas = nib.load(self.mni_atlas_label)
        self.atlas_labels = atlas.get_fdata()
        self.affine = atlas.affine
        self.header = atlas.header

    def _calc_weights(self):
        '''
        Calculate connectivity matrices based on chosen atlas and than calculate the
        weighted average to compute mean ADD in each label
        '''
        import numpy as np
        from weighted_tracts import nodes_by_index_general, nodes_labels_yeo7, nodes_labels_bna, non_weighted_con_mat_mega, weighted_con_mat_mega

        lab_labels_index, affine = nodes_by_index_general(self.folder_name, atlas=self.atlas_type)

        if 'yeo' in self.atlas_type:
            labels_headers, idx = nodes_labels_yeo7(self.idx_to_txt)
        elif 'bna' in self.atlas_type:
            labels_headers, idx = nodes_labels_bna(self.idx_to_txt)

        new_data, num_mat, grouping = non_weighted_con_mat_mega(self.streamlines, lab_labels_index, affine, idx, self.folder_name,
                                                          fig_type = self.atlas_type)

        add_mat = weighted_con_mat_mega(self.weight_name, grouping, idx, self.folder_name, fig_type = self.atlas_type)[1]

        mutual_mat = add_mat*num_mat

        #mutual_mat[mutual_mat<=0]= np.nan
        weights = np.sum(mutual_mat,1)/np.sum(num_mat,1)
        weights[np.isnan(weights)]=0
        weights_dict = {idx[i]:weights[i] for i in range(len(idx))}

        self.labels_weights = weights_dict

    def weight_gm_by_add(self):
        import numpy as np
        ''' for each voxel or brain area find all streamlines passes through it and average ADD of them'''
        self._load_mni_atlas()
        self._calc_weights()
        weighted_by_atlas = self.atlas_labels
        for i,weight in self.labels_weights.items():
            weighted_by_atlas[weighted_by_atlas==i+1] = weight

        self.weighted_by_atlas = np.asarray(weighted_by_atlas, dtype='float64')
        self.add_weighted_nii = nib.Nifti1Image(self.weighted_by_atlas, self.affine, self.header)

    def save_weighted_gm_mask(self,file_name):
        import nibabel as nib
        file_name = os.path.join(self.folder_name,file_name+'.nii')
        nib.save(self.add_weighted_nii, file_name)


if __name__ == '__main__':
    import glob
    weight_type = ['ADD','FA','MD']
    for subj_fol in glob.glob(f'F:\data\V7\TheBase4Ever\*{os.sep}'):
        atlas_name = 'yeo7_200'

        if not os.path.exists(os.path.join(subj_fol,'streamlines')):
            print('Could not find streamlines file')
            continue


        for wt in weight_type:
            file_name = f'{wt}_by_{atlas_name}'
            if os.path.exists(os.path.join(subj_fol, file_name + '.nii')):
                print(f'Done with \n {file_name} \n {subj_fol} \n')
            else:
                if 'FA' in wt:
                    subj_mask = GM_mask(subj_fol=subj_fol, atlas_name=atlas_name, weight_by='FA')
                elif 'MD' in wt:
                    subj_mask = GM_mask(subj_fol=subj_fol, atlas_name=atlas_name, weight_by='MD')
                else:
                    subj_mask = GM_mask(subj_fol=subj_fol, atlas_name=atlas_name)
                subj_mask.weight_gm_by_add()
                subj_mask.save_weighted_gm_mask(file_name=file_name)
                print(f'Done with \n {file_name} \n {subj_fol} \n')



