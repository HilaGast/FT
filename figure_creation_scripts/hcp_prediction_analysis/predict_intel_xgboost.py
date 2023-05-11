from figure_creation_scripts.plot_for_predictors_values import from_model_2_bar_fi
from HCP_network_analysis.prediction_model.predict_traits_by_networks import *
from HCP_network_analysis.hcp_cm_parameters import *
from figure_creation_scripts.plot_regression_model_results import model_heatmaps
from useful_tools.merge_dict_with_common_keys import *
from HCP_network_analysis.prediction_model.xgboost_linear_model import gradient_boost_model


show_graphs = False
atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
ncm = ncm_options[0]
atlas = atlases[0]
regularization = reg_options[0]
networks =['SomMot_LH', 'Vis_LH', 'Cont_LH', 'Default_LH', 'SalVentAttn_LH', 'DorsAttn_LH', 'Limbic_LH','SomMot_RH', 'Vis_RH', 'Cont_RH', 'Default_RH', 'SalVentAttn_RH', 'DorsAttn_RH', 'Limbic_RH','SomMot', 'Vis', 'Cont', 'Default', 'SalVentAttn', 'DorsAttn', 'Limbic', 'inter_network_LH', 'inter_network_RH']
traits = ['CogTotalComp_AgeAdj']
nw = len(weights)

model_params = {}
model_params['test_size'] = 0.2
model_params['alpha'] = 0  # 0.5 Lasso regularization alpha
model_params['eta'] = 0.05
model_params['n_estimators'] = 300
figs_folder = r'G:\data\V7\HCP\pca analysis results\Figs\xgboost - without regularization\subnetworks_pca20'

explained_var_table = pd.DataFrame(index = weights, columns = networks)
trait_name= traits[0]
all_var_table = pd.DataFrame(columns = ['component', 'explained_var','sub_network'])
model_r2_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_CVrmse_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_results_dict = {'R2': model_r2_table, 'RMSE': model_CVrmse_table}
model_allruns = dict()
for model_result in model_results_dict:
    model_allruns[model_result] = pd.DataFrame(columns=['value','trait','weight'])

feature_importance_table = pd.DataFrame(columns = networks)

for trait_name in traits:
    feature_importance_dict = {w: feature_importance_table for w in weights} #net_coeff_dict
    networks_pca_dict = dict()
    n_components_per_network_dict = dict()
    all_var_table_dict = dict()
    for weight_by in weights[:3]:
        print(trait_name, weight_by, regularization, ncm)
        all_var_table_dict[weight_by] = all_var_table
        if 'x' in weight_by:
            weight_by_couple = weight_by.split('x')
            networks_pca_dict[weight_by] = merge_dict_concatenate(networks_pca_dict[weight_by_couple[0]],
                                                                  networks_pca_dict[weight_by_couple[1]])
            n_components_per_network_dict[weight_by] = merge_dict_sum(n_components_per_network_dict[weight_by_couple[0]],
                                                                        n_components_per_network_dict[weight_by_couple[1]])
            all_var_table_dict[weight_by] = pd.concat([all_var_table_dict[weight_by_couple[0]], all_var_table_dict[weight_by_couple[1]]])
            all_var_table_dict[weight_by] = all_var_table_dict[weight_by].sort_values('sub_network')
            network_list = list(set(list(all_var_table_dict[weight_by]['sub_network'].sort_values())))
        else:
            subjects = glob.glob(
                f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
            connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
            networks_matrices, network_mask_vecs = from_whole_brain_to_networks(connectivity_matrices,
                                                                                atlas_index_labels, hemi_flag=True)
            networks_pca_dict[weight_by], n_components_per_network_dict[weight_by], explained_var_table, \
                all_var_table_dict[weight_by] = pca_for_each_network_different_number_of_components(networks_matrices,
                                                                                                    network_mask_vecs,
                                                                                                    explained_variance_th=0.2,
                                                                                                    explained_var_table=explained_var_table,
                                                                                                    weight_by=weight_by,
                                                                                                    all_var_table=
                                                                                                    all_var_table_dict[
                                                                                                        weight_by])
            network_list = list(set(list(all_var_table_dict[weight_by]['sub_network'])))
        trait_vector = load_trait_vector(trait_name, subjects)

        model, model_results_dict = gradient_boost_model(networks_pca_dict[weight_by],trait_vector,network_list,n_components_per_network_dict[weight_by],model_params, model_results_dict, trait_name, weight_by, figs_folder)

        print(f'Number of Predictors (after Lasso): {np.sum(np.abs(model.coef_) > 0)}')
        print(f'R2: {model_results_dict["R2"][trait_name][weight_by]}')
        print(f'RMSE: {model_results_dict["RMSE"][trait_name][weight_by]}')
        print('******************************')

        if show_graphs:
            from_model_2_bar_fi(model, network_list, all_var_table_dict[weight_by], n_components_per_network_dict[weight_by], trait_name, weight_by, ncm, atlas, figs_folder)







