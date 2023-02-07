import numpy as np
from HCP_network_analysis.hcp_cm_parameters import *
import matplotlib.pyplot as plt
import os, glob
import pandas as pd
from HCP_network_analysis.prediction_model.predict_traits_by_networks import create_all_subject_connectivity_matrices, \
    from_whole_brain_to_networks, pca_for_each_network_different_number_of_components, load_trait_vector, \
    linear_regression

num_iters = [5, 10, 50]#, 100, 500, 1000]
trait = 'CogTotalComp_AgeAdj'
atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
ncm = ncm_options[0]
atlas = atlases[0]
regularization = reg_options[1]
weight_by = weights[2]
variation_explained = [0.2, 0.4]
figs_folder = r'G:\data\V7\HCP\pca analysis results\Figs\Intelligence multiple matrices - stability over iterations'
all_var_table = pd.DataFrame(columns = ['component', 'explained_var','sub_network'])
explained_var_table = pd.DataFrame(index = weights, columns = ['SomMot', 'Vis', 'Cont', 'Default', 'SalVentAttn', 'DorsAttn', 'Limbic', 'inter_network'])
model_r2_table = pd.DataFrame(index = weights, columns = [trait], dtype='float')
model_F_table = pd.DataFrame(index = weights, columns = [trait], dtype='float')
model_p_table = pd.DataFrame(index = weights, columns = [trait], dtype='float')
model_CVrmse_table = pd.DataFrame(index = weights, columns = [trait], dtype='float')
model_accuracy_table = pd.DataFrame(index = weights, columns = [trait], dtype='float')
model_results_dict = {'R2_adj': model_r2_table, 'F': model_F_table, 'p-value': model_p_table,'CV(RMSE)': model_CVrmse_table,'Accuracy': model_accuracy_table}
for ve in variation_explained:
    print(f'** {ve} var explained **')
    mean_r = []
    std_r = []
    for ni in num_iters:
        print(f'* {ni} iterations')
        adjusted_r2=[]
        for run in range(1, ni + 1):
            subjects = glob.glob(
                f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
            connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
            networks_matrices, network_mask_vecs = from_whole_brain_to_networks(connectivity_matrices,
                                                                                atlas_index_labels)
            networks_pca_dict, n_components_per_network_dict, explained_var_table, \
                all_var_table = pca_for_each_network_different_number_of_components(networks_matrices,
                                                                                                    network_mask_vecs,
                                                                                                    explained_variance_th=ve,
                                                                                                    explained_var_table=explained_var_table,
                                                                                                    weight_by=weight_by,
                                                                                                    all_var_table=all_var_table)
            network_list = list(set(list(all_var_table['sub_network'])))
            trait_vector = load_trait_vector(trait, subjects)
            model, model_results_dict, mask_vec = linear_regression(networks_pca_dict, trait_vector,
                                                                    network_list,
                                                                    n_components=n_components_per_network_dict,
                                                                    model_results_dict=model_results_dict,
                                                                    weight_by=weight_by, trait_name=trait,
                                                                    lasso_regularized=True, alpha=1)
            adjusted_r2.append(model_results_dict['R2_adj'])
        mean_r.append(np.nanmean(adjusted_r2))
        std_r.append(np.nanstd(adjusted_r2))
        print(f'{mean_r} , {std_r}')

    plt.errorbar(range(len(num_iters)),mean_r,yerr=std_r,linestyle='None',fmt='o',capsize=3)
    plt.title(f'Mean adjusted $r^{2}$ \n'
              f'{int(ve*100)}% variation explained')
    plt.xticks(range(len(num_iters)), num_iters)
    plt.xlabel('# iterations')
    plt.ylabel('Mean adjusted $r^{2}$')
    plt.savefig(f'{figs_folder}{os.sep}{weight_by}_{int(ve*100)}_var_explained.png')
    plt.show()