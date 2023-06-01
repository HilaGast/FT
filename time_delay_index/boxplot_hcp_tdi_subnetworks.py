from scipy.stats import ttest_ind

from Tractography.group_analysis import create_all_subject_connectivity_matrices, norm_matrices
import glob, os
from yeo_sub_networks.from_wb_mat_to_subs import from_whole_brain_to_networks
import numpy as np

def compare_within_between_networks(networks_matrices):

    val_within_between = dict()
    within_vals=[]
    between_vals=[]
    for k,v in networks_matrices.items():
        v[v==0]=np.nan
        for s in range(networks_matrices[k].shape[-1]):
            if 'inter_network' in k:
                between_vals.append(np.nanmedian(v[:,:,s]))
            else:
                within_vals.append(np.nanmedian(v[:,:,s]))
    val_within_between['within'] = within_vals
    val_within_between['between'] = between_vals
    t, p = ttest_ind(val_within_between['within'],val_within_between['between'])
    print(f't={t}, p={p}')
    return val_within_between

def compare_networks(networks_matrices, network_names):
    if networks_matrices[network_names[0]].shape[-1] == 1:
        val_per_net = dict()
        for k, v in networks_matrices.items():
            v[v == 0] = np.nan
            val_per_net[k] = v[~np.isnan(v)]
        return val_per_net
    val_per_net = dict()
    for k,v in networks_matrices.items():
        v[v==0]=np.nan
        val_per_net[k] = []
        for s in range(networks_matrices[k].shape[-1]):
            val_per_net[k].append(np.nanmedian(v[: ,:,s]))
    return val_per_net

def boxplot_8networks(net_dict,hemi=''):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    try:
        df = pd.DataFrame.from_dict(net_dict)
        df = df.melt(var_name='Network', value_name='Scaled TDI [0-1]')
        #df = df.melt(var_name='Network', value_name='Pearson r')
        df = df.sort_values(by=['Network'])
        plt.figure(figsize=(14, 10))
        sns.set_theme(style="whitegrid", palette="Spectral",font_scale=1.5)
        ax = sns.boxplot(x="Network", y="Scaled TDI [0-1]", data=df)
        #ax = sns.boxplot(x="Network", y="Pearson r", data=df)
        plt.title(f'{hemi}')
        plt.show()
    except ValueError:
        sns.set(style="whitegrid")
        if hemi == '':
            ax = sns.boxplot([net_dict['Vis'], net_dict['SomMot'], net_dict['DorsAttn'], net_dict['SalVentAttn'],net_dict['Limbic'], net_dict['Cont'], net_dict['Default'], net_dict['inter_network']], palette="Set3")
        else:
            ax = sns.boxplot([net_dict[f'Vis_{hemi}'], net_dict[f'SomMot_{hemi}'], net_dict[f'DorsAttn_{hemi}'], net_dict[f'SalVentAttn_{hemi}'],net_dict[f'Limbic_{hemi}'], net_dict[f'Cont_{hemi}'], net_dict[f'Default_{hemi}'], net_dict[f'inter_network_{hemi}']], palette="Set3")
        plt.show()

def keep_one_hemi(networks_matrices, hemi):
    one_hemi_matrices = dict()
    for k,v in networks_matrices.items():
        if hemi in k:
            one_hemi_matrices[k] = v
    return one_hemi_matrices


if __name__ == '__main__':
    mat_type = 'time_th3'
    atlas_index_labels = r'G:\data\atlases\yeo\yeo7_100\index2label.txt'
    atlas = 'yeo7_100'
    subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
    subjects=[]
    # all subjects:
    for s in subj_list:
        tdi_file_name = f'{s}cm{os.sep}{atlas}_time_th3_Org_SC_cm_ord.npy'
        fmri_file_name = f'{s}cm{os.sep}{atlas}_fmri_Org_SC_cm_ord.npy'
        if os.path.exists(tdi_file_name) and os.path.exists(fmri_file_name):
            subjects.append(fmri_file_name)

    #subjects = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}yeo7_100_{mat_type}_Org_SC_cm_ord.npy')
    # median mat:
    #subjects = [rf'G:\data\V7\HCP\cm\median_yeo7_100_{mat_type}_Org_SC.npy']

    connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
    #norm_cm = norm_matrices(connectivity_matrices, norm_type='scaling')
    norm_cm = connectivity_matrices
    networks_matrices, network_mask_vecs = from_whole_brain_to_networks(norm_cm, atlas_index_labels,
                                                                    hemi_flag=True)
    networks_matrices = keep_one_hemi(networks_matrices, 'LH')
    network_names = list(networks_matrices.keys())
    val_per_net = compare_networks(networks_matrices, network_names)
    boxplot_8networks(val_per_net, hemi='LH')

