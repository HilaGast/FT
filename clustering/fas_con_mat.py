from FT.weighted_tracts import *
from os.path import join as pjoin
from dipy.io.streamline import load_trk


if __name__ == '__main__':
    subj = all_subj_folders
    names = all_subj_names
    index_to_text_file = r'C:\Users\hila\data\megaatlas\megaatlas2nii.txt'
    fig_types = ['SLF','AF']
    weight_by='1.5_2_AxPasi5'
    for s, n in zip(subj, names):
        folder_name = subj_folder + s
        dir_name = folder_name + '\streamlines'
        bvec_file = load_dwi_files(folder_name)[6]
        lab_labels_index = nodes_by_index_mega(folder_name)[0]
        file_list = os.listdir(dir_name)
        file_list = [l for l in file_list if 'mct001rt20_4d' in l]
        for fig_type in fig_types:
            for file in file_list:
                if fig_type in file and '.trk' in file and '_L' in file:
                    fascicle_file_name_l = pjoin(dir_name, file)
                    s_l = load_trk(fascicle_file_name_l, "same", bbox_valid_check=False)
                    s_l = s_l.streamlines
                elif fig_type in file and '.trk' in file and '_R' in file:
                    fascicle_file_name_r = pjoin(dir_name, file)
                    s_r = load_trk(fascicle_file_name_r, "same", bbox_valid_check=False)
                    s_r = s_r.streamlines
            s_l.extend(s_r)
            streamlines = s_l

            weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type=fig_type,
                                              weight_by=weight_by)
