from HCP_network_analysis.prediction_model.predict_traits_wb import whole_brain_matrix
from figure_creation_scripts.plot_for_predictors_values import from_model_2_bar, plot_coeff_over_iterations
from HCP_network_analysis.prediction_model.predict_traits_by_networks import *
from HCP_network_analysis.hcp_cm_parameters import *
from figure_creation_scripts.plot_regression_model_results import *
from useful_tools.merge_dict_with_common_keys import *
from HCP_network_analysis.prediction_model.xgboost_linear_model import *

atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
ncm = ncm_options[0]
atlas = atlases[0]
regularization = reg_options[0]
#networks =['SomMot_LH', 'Vis_LH', 'Cont_LH', 'Default_LH', 'SalVentAttn_LH', 'DorsAttn_LH', 'Limbic_LH','SomMot_RH', 'Vis_RH', 'Cont_RH', 'Default_RH', 'SalVentAttn_RH', 'DorsAttn_RH', 'Limbic_RH','SomMot', 'Vis', 'Cont', 'Default', 'SalVentAttn', 'DorsAttn', 'Limbic', 'inter_network_LH', 'inter_network_RH']
traits = ['CogTotalComp_AgeAdj']

etas = [0.005, 0.01, 0.05, 0.1]
n_est = [10, 50, 100, 200, 300, 500]
alphas = [1]
model_params = {}
for alpha in alphas:
    model_params['alpha'] = alpha  # Lasso regularization alpha

    explained_var_table = pd.DataFrame(index=weights, columns=['wb'])
    trait_name = traits[0]
    all_var_table = pd.DataFrame(columns=['component', 'explained_var', 'sub_network'])

    all_var_table_dict = dict()
    for weight_by in weights[:4]:
        n_components_per_network_dict = dict()
        networks_pca_dict = dict()
        all_var_table_dict[weight_by] = all_var_table

        subjects = glob.glob(
            f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
        trait_vector = load_trait_vector(trait_name, subjects)
        connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
        networks_matrices, network_mask_vecs = whole_brain_matrix(connectivity_matrices)
        networks_pca_dict[weight_by], n_components_per_network_dict[weight_by], explained_var_table, \
            all_var_table_dict[weight_by] = pca_for_each_network_different_number_of_components(networks_matrices,
                                                                                                network_mask_vecs,
                                                                                                explained_variance_th=0.3,
                                                                                                explained_var_table=explained_var_table,
                                                                                                weight_by=weight_by,
                                                                                                all_var_table=
                                                                                                all_var_table_dict[
                                                                                                    weight_by])
        network_list = list(set(list(all_var_table_dict[weight_by]['sub_network'])))

        train_gradient_boost_model_cv_logloss(networks_pca_dict[weight_by], trait_vector, network_list,
                                              n_components_per_network_dict[weight_by],
                                              model_params, etas, n_est, weight_by)
