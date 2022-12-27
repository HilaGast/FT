import pandas as pd

from figure_creation_scripts.plot_for_predictors_values import from_model_2_bar
from HCP_network_analysis.predict_traits_by_networks import *
from HCP_network_analysis.hcp_cm_parameters import *
from figure_creation_scripts.plot_for_pca_components import *
from figure_creation_scripts.plot_regression_model_results import *
from useful_tools.merge_dict_with_common_keys import *

show_graphs = True
atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
ncm = ncm_options[0]
atlas = atlases[0]
regularization = reg_options[1]


traits = ['CogTotalComp_AgeAdj', 'CogFluidComp_AgeAdj', 'CogCrystalComp_AgeAdj']

explained_var_table = pd.DataFrame(index = weights, columns = ['SomMot', 'Vis', 'Cont', 'Default', 'SalVentAttn', 'DorsAttn', 'Limbic', 'inter_network'])
model_r2_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_F_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_p_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_CVrmse_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_accuracy_table = pd.DataFrame(index = weights, columns = traits, dtype='float')

model_results_dict = {'R2_adj': model_r2_table, 'F': model_F_table, 'p-value': model_p_table,'CV(RMSE)': model_CVrmse_table,'Accuracy': model_accuracy_table}
figs_folder = r'G:\data\V7\HCP\pca analysis results\Figs\Intelligence multiple matrices - multi runs'
all_var_table = pd.DataFrame(columns = ['component', 'explained_var','sub_network'])
for trait_name in traits:
    networks_pca_dict = dict()
    n_components_per_network_dict = dict()
    all_var_table_dict = dict()
    for weight_by in weights:
        all_var_table_dict[weight_by] = all_var_table
        if 'x' in weight_by:
            weight_by_couple = weight_by.split('x')
            networks_pca_dict[weight_by] = merge_dict_concatenate(networks_pca_dict[weight_by_couple[0]], networks_pca_dict[weight_by_couple[1]])
            n_components_per_network_dict[weight_by] = merge_dict_sum(n_components_per_network_dict[weight_by_couple[0]], n_components_per_network_dict[weight_by_couple[1]])
            all_var_table_dict[weight_by] = pd.concat([all_var_table_dict[weight_by_couple[0]], all_var_table_dict[weight_by_couple[1]]])
            all_var_table_dict[weight_by] = all_var_table_dict[weight_by].sort_values('sub_network')
            network_list = list(set(list(all_var_table_dict[weight_by]['sub_network'].sort_values())))

        else:
            subjects = glob.glob(
                f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
            connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
            networks_matrices, network_mask_vecs = from_whole_brain_to_networks(connectivity_matrices, atlas_index_labels)
            networks_pca_dict[weight_by], n_components_per_network_dict[weight_by], explained_var_table, all_var_table_dict[weight_by] = pca_for_each_network_different_number_of_components(networks_matrices,
                                                                                                     network_mask_vecs,
                                                                                                     explained_variance_th=0.2,explained_var_table=explained_var_table, weight_by=weight_by, all_var_table = all_var_table_dict[weight_by])
            network_list = list(set(list(all_var_table_dict[weight_by]['sub_network'])))

        print(weight_by, regularization, ncm)
        print(f'*********** \n alpha: 1.0 \n ***********')


        trait_vector = load_trait_vector(trait_name, subjects)
        model, model_results_dict, mask_vec = linear_regression(networks_pca_dict[weight_by], trait_vector, network_list,
                                                                    n_components=n_components_per_network_dict[weight_by],
                                                                    model_results_dict=model_results_dict,
                                                                    weight_by=weight_by, trait_name=trait_name,
                                                                    lasso_regularized=True, alpha=1)
        print(f'Number of Predictors (after Lasso): {np.sum(model.params>0)}')
        print(f'R2_adj: {model_results_dict["R2_adj"][trait_name][weight_by]}')
        print(f'p-value: {model_results_dict["p-value"][trait_name][weight_by]}')
        print(f'CV(RMSE): {100*model_results_dict["CV(RMSE)"][trait_name][weight_by]}%')
        print(f'Accuracy: {model_results_dict["Accuracy"][trait_name][weight_by]}')
        print('***********')

        #all_var_table = all_var_table[mask_vec[1:]]
        if model.f_pvalue < 0.05 and show_graphs:
            sum_table = from_model_2_bar(model, network_list, all_var_table_dict[weight_by], n_components_per_network_dict[weight_by], trait_name, weight_by, ncm, regularization, atlas, figs_folder, show_all=False)[0]

        #if show_graphs:
            #plot_components_after_lasso(all_var_table_dict[weight_by], trait_name, weight_by, mask_vec, figs_folder)

#if show_graphs:
    #plot_components_before_lasso(explained_var_table, '', figs_folder)

if show_graphs:
    label = f'{atlas}_{trait_name}'
    model_heatmaps(model_results_dict, weights, figs_folder, label=label)



