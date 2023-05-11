import pandas as pd

from HCP_network_analysis.prediction_model.predict_traits_by_networks import *
from HCP_network_analysis.hcp_cm_parameters import *
from HCP_network_analysis.prediction_model.xgboost_linear_model import gradient_boost_permutation_test

show_graphs = True
atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
ncm = ncm_options[0]
atlas = atlases[0]
regularization = reg_options[0]
networks =['SomMot_LH', 'Vis_LH', 'Cont_LH', 'Default_LH', 'SalVentAttn_LH', 'DorsAttn_LH', 'Limbic_LH','SomMot_RH', 'Vis_RH', 'Cont_RH', 'Default_RH', 'SalVentAttn_RH', 'DorsAttn_RH', 'Limbic_RH','SomMot', 'Vis', 'Cont', 'Default', 'SalVentAttn', 'DorsAttn', 'Limbic', 'inter_network_LH', 'inter_network_RH']
to_remove = ['None','SomMot', 'Vis', 'Cont', 'Default', 'SalVentAttn', 'DorsAttn', 'Limbic', 'inter_network','LH', 'RH']
trait_name = 'CogTotalComp_AgeAdj'
nw = len(weights)

model_params = {}
model_params['test_size'] = 0.2
model_params['alpha'] = 1  # 0.5 Lasso regularization alpha
model_params['eta'] = 0.05
model_params['n_estimators'] = 200

figs_folder = r'G:\data\V7\HCP\pca analysis results\Figs\xgboost_results - PCA20 - Leave_one_out'
for weight_by in weights[:3]:
    leave_one_out_table = pd.DataFrame(columns=to_remove)
    for loo in to_remove:
        networks_to_keep = [x for x in networks if loo not in x]

        explained_var_table = pd.DataFrame(index=weights, columns=networks_to_keep)
        all_var_table = pd.DataFrame(columns=['component', 'explained_var', 'sub_network'])
        model_r2_table = pd.DataFrame(index=weights, columns=[trait_name], dtype='float')
        model_CVrmse_table = pd.DataFrame(index=weights, columns=[trait_name], dtype='float')
        model_results_dict = {'R2': model_r2_table, 'RMSE': model_CVrmse_table}
        model_allruns = dict()
        for model_result in model_results_dict:
            model_allruns[model_result] = pd.DataFrame(columns=['value', 'trait', 'weight'])
        networks_pca_dict = dict()
        n_components_per_network_dict = dict()
        all_var_table_dict = dict()

        print(trait_name, weight_by, regularization, ncm, f'Leave one out: {loo}')
        all_var_table_dict[weight_by] = all_var_table


        subjects = glob.glob(
                f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
        connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
        networks_matrices, network_mask_vecs = from_whole_brain_to_networks(connectivity_matrices,
                                                                                atlas_index_labels, hemi_flag=True)
        networks_matrices = {key: networks_matrices[key] for key in networks_to_keep}
        network_mask_vecs = {key: network_mask_vecs[key] for key in networks_to_keep}
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

        all_r = gradient_boost_permutation_test(networks_pca_dict[weight_by], trait_vector, network_list,
                                                n_components_per_network_dict[weight_by], model_params, trait_name,
                                                weight_by, figs_folder, n=100)
        leave_one_out_table[loo] = all_r

    leave_one_out_table.to_excel(rf'G:\data\V7\HCP\pca analysis results\Figs\xgboost_results - PCA20 - Leave_one_out\leave_one_out_table_{weight_by}_N100.xlsx')




