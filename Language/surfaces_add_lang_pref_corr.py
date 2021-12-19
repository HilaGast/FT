
from parcellation.nodes_add_correlation_to_age import *
from parcellation.add_weighted_gm_mask import *


def add_lang_pref(subj_main_folder, atlas_main_folder, atlas_name):
    table1 = SubjTable(r'C:\Users\Admin\Desktop\Language\Subject list - Language.xlsx', 'Sheet1')

    wos1 = []
    lws = []
    lwd = []

    volume_type = 'ADD'
    vol_mat, mni_atlas_file_name, idx, subj_idx = volume_based_var(atlas_name, volume_type, atlas_main_folder,
                                                                   subj_main_folder)
    for sub in glob.glob(f'{subj_main_folder}{os.sep}*{os.sep}'):
        sn = sub.split(os.sep)[-2]
        wos1.append(table1.find_value_by_scan_Language('Word Order Score 1', sn))
        lws.append(table1.find_value_by_scan_Language('Learning words slope', sn))
        lwd.append(table1.find_value_by_scan_Language('Learning words dist', sn))

    r,p = corr_stats(vol_mat, wos1)

    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_WOS1_r_'+atlas_name, subj_main_folder)
    r_th, p_corr = multi_comp_correction(r, p)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r_th,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_WOS1_th_r_'+atlas_name, subj_main_folder)

    r, p = corr_stats(vol_mat, lws)

    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LWS_r_' + atlas_name, subj_main_folder)
    r_th, p_corr = multi_comp_correction(r, p)
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r_th, idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LWS_th_r_' + atlas_name, subj_main_folder)

    r, p = corr_stats(vol_mat, lwd)

    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LWD_r_' + atlas_name, subj_main_folder)
    r_th, p_corr = multi_comp_correction(r, p)
    weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r_th, idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LWD_th_r_' + atlas_name, subj_main_folder)


def add_lang_pref_by_yeo_net(subj_main_folder, atlas_main_folder, atlas_name):
    from Ageing.correlation_with_age_by_yeonet import corr_spec_net

    table1 = SubjTable(r'C:\Users\Admin\Desktop\Language\Subject list - Language.xlsx', 'Sheet1')

    wos1 = []
    lws = []
    lwd = []

    volume_type = 'ADD'
    vol_mat, mni_atlas_file_name, idx, subj_idx = volume_based_var(atlas_name, volume_type, atlas_main_folder,
                                                                   subj_main_folder)
    for sub in glob.glob(f'{subj_main_folder}{os.sep}*{os.sep}'):
        sn = sub.split(os.sep)[-2]
        wos1.append(table1.find_value_by_scan_Language('Word Order Score 1', sn))
        lws.append(table1.find_value_by_scan_Language('Learning words slope', sn))
        lwd.append(table1.find_value_by_scan_Language('Learning words dist', sn))

    net_names = ['vis','sommot','dorsattn','SalVentAttn','limbic','cont','default']


    for net_name in net_names:
        r,p = corr_stats(vol_mat, wos1)
        r, p, r_th = corr_spec_net(r, p, net_name)
        weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
        save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_WOS1_r_{atlas_name}_{net_name}',
                    subj_main_folder)
        weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r_th, idx)
        save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_WOS1_th_r_{atlas_name}_{net_name}',
                    subj_main_folder)


    for net_name in net_names:
        r, p = corr_stats(vol_mat, lws)
        r, p, r_th = corr_spec_net(r, p, net_name)
        weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
        save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LWS_r_{atlas_name}_{net_name}',
                    subj_main_folder)
        weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r_th, idx)
        save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LWS_th_r_{atlas_name}_{net_name}',
                    subj_main_folder)

    for net_name in net_names:
        r, p = corr_stats(vol_mat, lwd)
        r, p, r_th = corr_spec_net(r, p, net_name)
        weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
        save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LWD_r_{atlas_name}_{net_name}',
                    subj_main_folder)
        weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r_th, idx)
        save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_LWD_th_r_{atlas_name}_{net_name}',
                    subj_main_folder)


if __name__ == '__main__':
    import glob
    atlas_main_folder = r'C:\Users\Admin\my_scripts\aal\yeo'
    subj_main_folder = r'C:\Users\Admin\Desktop\Language'
    weight_type = ['ADD','FA','MD']
    atlas_name = 'yeo7_200'
    for subj_fol in glob.glob(f'{subj_main_folder}{os.sep}*{os.sep}'):

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

    add_lang_pref(subj_main_folder, atlas_main_folder, atlas_name)

    add_lang_pref_by_yeo_net(subj_main_folder, atlas_main_folder, atlas_name)





