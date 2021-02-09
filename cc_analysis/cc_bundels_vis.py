
from weighted_tracts import *
from single_fascicle_vizualization import create_streamline_dict, load_mat_of_median_vals,show_fascicles_wholebrain


def choose_cc_bundle(streamlines,affine,folder_name,part):
    from dipy.tracking import utils
    from dipy.tracking.streamline import Streamlines

    file_list = os.listdir(folder_name)
    for file in file_list:
        if part in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            cc_mask_mat = mask_img.get_data()

    mask_include = cc_mask_mat == 1
    masked_streamlines = utils.target(streamlines, affine, mask_include)
    masked_streamlines = Streamlines(masked_streamlines)

    return masked_streamlines


def clean_non_cc(grouping):
    clean_grouping = {}
    for pair, tracts in grouping.items():
        if pair[0] == 0 or pair[1] == 0:
            continue
        else:
            if pair[0] in list(np.arange(1,101)) and pair[1] in list(np.arange(101,201)):
                clean_grouping[pair] = tracts
            if pair[1] in list(np.arange(1,101)) and pair[0] in list(np.arange(101,201)):
                clean_grouping[pair] = tracts
    return clean_grouping


if __name__ == '__main__':
    from dipy.tracking.streamline import Streamlines
    from scipy.stats import f_oneway

    main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep'
    folder_name = main_folder+all_subj_folders[0]
    n = all_subj_names[0]
    nii_file = load_dwi_files(folder_name)[5]


    tract_path = folder_name+r'\streamlines'+n+'_wholebrain_3d_plus_new.trk'
    streamlines = load_ft(tract_path, nii_file)

    lab_labels_index, affine = nodes_by_index_mega(folder_name)
    mat_medians = load_mat_of_median_vals(mat_type='w_plus')
    index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlas2nii.txt'
    idx = nodes_labels_mega(index_to_text_file)[1]
    id = np.argsort(idx)
    mat_medians = mat_medians[id]
    mat_medians = mat_medians[:,id]

    vec_vols_all = []
    s_list_all=[]

    #masks = ['genu','bodya','bodym','bodyp','splenium']
    masks = ['genu', 'body', 'splenium']
    vols_list = []
    for m in masks:
        masked_streamlines = choose_cc_bundle(streamlines, affine, folder_name, m)
        streamline_dict = create_streamline_dict(masked_streamlines, lab_labels_index, affine)
        streamline_dict = clean_non_cc(streamline_dict)
        vec_vols = []
        s_list = []
        for i in range(id.__len__()):
            for j in range(i + 1):
                edge_s_list = []
                if (i + 1, j + 1) in streamline_dict and mat_medians[i, j] > 1:
                    edge_s_list += streamline_dict[(i + 1, j + 1)]
                if (j + 1, i + 1) in streamline_dict and mat_medians[i, j] > 1:
                    edge_s_list += streamline_dict[(j + 1, i + 1)]

                edge_vec_vols = [mat_medians[i, j]] * edge_s_list.__len__()
                s_list = s_list + edge_s_list
                vec_vols = vec_vols + edge_vec_vols

        s_list_all += s_list
        vec_vols_all = vec_vols_all+[np.nanmean(vec_vols)] * s_list.__len__()
        vols_list.append(list(vec_vols))
        print(m ,'   ' ,np.nanmean(vec_vols))
    s_list_all = Streamlines(s_list_all)
    #show_fascicles_wholebrain(s_list_all, vec_vols_all,folder_name)
    #show_fascicles_wholebrain(s_list_all, vec_vols_all,folder_name)
    st, pval = f_oneway(vols_list[0], vols_list[1], vols_list[2])
    print(f'F = {st} , pval = {pval}')




