from weighted_tracts import *


subj_folder = r''
all_folders = os.listdir(subj_folder)
all_subj_folders = list()
all_subj_names = list()
for subj in all_folders[::]:
    name = subj.split('_')
    name = '/'+name[3]
    name = name.replace('/',os.sep)
    all_subj_names.append(name)
    subj = '/' + subj
    subj = subj.replace('/', os.sep)
    all_subj_folders.append(subj)

for n,s in zip(all_subj_names,all_subj_names):
    folder_name = subj_folder + s
    dir_name = folder_name + '\streamlines'


    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
    csd_fit = create_csd_model(data, gtab, white_matter, sh_order=6)
    fa, classifier = create_fa_classifier(gtab, data, white_matter)
    lab_labels_index = nodes_by_index_general(folder_name, atlas='yeo7_200')[0]
    seeds = create_seeds(folder_name, lab_labels_index, affine, use_mask=False, mask_type='cc', den=4)
    streamlines = create_streamlines(csd_fit, classifier, seeds, affine)
    save_ft(folder_name, n, streamlines, nii_file, file_name="_wholebrain_4d_labmask.trk")
    idx = nodes_labels_yeo7(index_to_text_file)[1]
    weighted_connectivity_matrix_mega(streamlines, folder_name, bvec_file, fig_type='wholebrain_4d_labmask_yeo7_200_FA',
                                  weight_by='_FA')