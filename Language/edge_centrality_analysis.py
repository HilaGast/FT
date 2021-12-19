import glob,os
from reading_from_xls.read_details_from_subject_table import *
from calc_corr_statistics.pearson_r_calc import *
from parcellation.group_weight import atlas_and_idx, weight_atlas_by_add, save_as_nii
from network_analysis.specific_functional_yeo_network import network_id_list
from network_analysis.edge_betweeness_centrality_mat import mat_ebc

if __name__ == "__main__":
    main_subj_folders = r'C:\Users\Admin\Desktop\Language'
    atlas_main_folder = r'C:\Users\Admin\my_scripts\aal\yeo'
    atlas_type = 'yeo7_200'
    #id1 = network_id_list(network_type='salventattn')-1
    #id2 = network_id_list(network_type='default')-1
    #idx = id1+id2
    atlas_labels, mni_atlas_label, idx = atlas_and_idx(atlas_type, atlas_main_folder)


    table1 = SubjTable(r'C:\Users\Admin\Desktop\Language\Subject list - Language.xlsx', 'Sheet1')
    wos1 = []
    lws = []
    n_subj= 0
    ebc_num = np.zeros((len(idx),len(idx),1))
    ebc_add = np.zeros((len(idx),len(idx),1))

    for sub in glob.glob(f'{main_subj_folders}{os.sep}*{os.sep}'):
        sn = sub.split(os.sep)[-2]
        num_mat_name = sub + 'non-weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
        if os.path.exists(num_mat_name):
            n_subj+=1
            num_mat = np.load(num_mat_name)
            ebc_numi = np.tril(mat_ebc(num_mat).reshape((len(idx),len(idx),1)))
            ebc_num = np.concatenate((ebc_num,ebc_numi),axis=-1)



            add_mat_name = sub + 'weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
            add_mat = np.load(add_mat_name)
            ebc_addi = np.tril(mat_ebc(add_mat).reshape((len(idx),len(idx),1)))
            ebc_add = np.concatenate((ebc_add,ebc_addi),axis=-1)


            wos1.append(table1.find_value_by_scan_Language('Word Order Score 1', sn))
            lws.append(table1.find_value_by_scan_Language('Learning words slope', sn))

    ebc_num = ebc_num[:,:,1:]
    ebc_add = ebc_add[:,:,1:]


    ebc_wos1_num = np.zeros((ebc_num.shape[0],ebc_num.shape[1]))
    ebc_lws_num = np.zeros((ebc_num.shape[0],ebc_num.shape[1]))

    ebc_wos1_add = np.zeros((ebc_num.shape[0],ebc_num.shape[1]))
    ebc_lws_add = np.zeros((ebc_num.shape[0],ebc_num.shape[1]))

    for r in range(0,ebc_num.shape[0]):
        for c in range(0,r):

            vec = ebc_num[r,c,:]

            if np.count_nonzero(vec)>=n_subj/2:
                rp,p = calc_corr(wos1,list(vec), remove_outliers=True)
                ebc_wos1_num[r,c] = rp
                rp,p = calc_corr(lws,list(vec), remove_outliers=True)
                ebc_lws_num[r,c] = rp

            else:
                ebc_wos1_num[r, c] = 0
                ebc_lws_num[r, c] = 0

    for r in range(0,ebc_add.shape[0]):
        for c in range(0, r):

            vec = ebc_add[r, c, :]

            if np.count_nonzero(vec) >= n_subj / 2:
                rp, p = calc_corr(wos1, list(vec), remove_outliers=True)
                ebc_wos1_add[r, c] = rp
                rp, p = calc_corr(lws, list(vec), remove_outliers=True)
                ebc_lws_add[r, c] = rp

            else:
                ebc_wos1_add[r, c] = 0
                ebc_lws_add[r, c] = 0

    import matplotlib.pyplot as plt

    plt.imshow(ebc_wos1_num)
    plt.show()

    plt.imshow(ebc_lws_num)
    plt.show()

    plt.imshow(ebc_wos1_add)
    plt.show()

    plt.imshow(ebc_lws_add)
    plt.show()







