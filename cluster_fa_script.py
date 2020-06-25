from clustering.cluster_fascicles import *

main_folder = subj_folder
show_cluster = True
fascicle = 'SLF_L_mct001rt20'
methods = ['agglomerative', 'kmeans']
method = methods[1]
for fol, n in zip(all_subj_folders, all_subj_names):
    folder_name = main_folder + fol
    if not any(fi.startswith(f'rdti_fa') for fi in os.listdir(folder_name)):
        print('Moving on!')
        continue

    print(f'++++++Starting to compute models for {n}++++++')
    nii_file = load_dwi_files(folder_name)[5]
    file_list = os.listdir(folder_name + r'\streamlines')
    for file in file_list:
        if fascicle in file and '.trk' in file:
            fascicle_file_name = pjoin(folder_name + r'\streamlines', file)
            s_list = load_trk(fascicle_file_name, "same", bbox_valid_check=False)
            masked_streamlines = s_list.streamlines
            break
    streamlines, vec_vols = streamline_mean_fascicle_value_weighted(folder_name, n, nii_file, fascicle,
                                                                    masked_streamlines, weight_by='rdti_fa')
    dist_method = 'mam'
    tracts_num = streamlines.__len__()

    #X = clustering_input(dist_method, tracts_num, streamlines, vec_vols)

    # show_23456_groups(method,streamlines,folder_name,X,fascicle)

    g = [3]
    for i in g:
        # model = compute_clustering_model(method, X, i)
        #save_model(model, i, folder_name, method, fascicle)
        model = load_model(i,folder_name,method, fascicle)
        weighted_clusters(model,streamlines,vec_vols, folder_name, file_name = 'clustered_'+fascicle+'_'+str(n))
