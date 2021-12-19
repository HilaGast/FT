from network_analysis.specific_functional_yeo_network import *
from parcellation.nodes_add_correlation_to_age import *


def corr_spec_net(r, p , net_name):
    from statsmodels.stats.multitest import multipletests as fdr
    import copy
    id_net = network_id_list(network_type = net_name)
    mask = np.ones(len(r), bool)
    mask[np.asarray(id_net)-1] = False
    r = np.asarray(r)
    p = np.asarray(p)
    r[mask] = 0
    p[mask] = 0
    for_comp = [p>0]
    p_corr_fc = fdr(p[for_comp],0.05,'fdr_bh')[1]
    p_corr = p
    p_corr[for_comp] = p_corr_fc

    r_th = np.asarray(copy.deepcopy(r))
    r_th[np.asarray(p_corr)>0.05]=0
    r_th = list(r_th)

    return r,p,r_th


if __name__ == '__main__':
    subj_main_folder = 'F:\data\V7\TheBase4Ever'
    atlas_type = 'yeo7_200'
    atlas_main_folder = r'C:\Users\Admin\my_scripts\aal\yeo'
    volume_type = 'ADD'
    vol_mat, mni_atlas_file_name, idx, subj_idx = volume_based_var(atlas_type,volume_type, atlas_main_folder, subj_main_folder)
    ages = age_var(subj_main_folder, subj_idx)
    r,p = corr_stats(vol_mat, ages)

    net_names = ['vis','sommot','dorsattn','SalVentAttn','limbic','cont','default']
    for net_name in net_names:
        r, p, r_th = corr_spec_net(r, p, net_name)
        weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r, idx)
        save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_AGE_r_{atlas_type}_{net_name}',
                    subj_main_folder)
        weighted_by_atlas, weights_dict = weight_atlas_by_add(mni_atlas_file_name, r_th, idx)
        save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'{volume_type}_AGE_th_r_{atlas_type}_{net_name}',
                    subj_main_folder)
