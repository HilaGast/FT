from HCP_network_analysis.prediction_model.predict_trait_by_dist_mat import from_whole_brain_to_dist_net
from HCP_network_analysis.prediction_model.predict_trait_by_hemisphere import from_whole_brain_to_hemispheres
from figure_creation_scripts.plot_for_predictors_values import from_model_2_bar, plot_coeff_over_iterations
from HCP_network_analysis.prediction_model.predict_traits_by_networks import *
from HCP_network_analysis.hcp_cm_parameters import *
from figure_creation_scripts.plot_regression_model_results import *
from useful_tools.merge_dict_with_common_keys import *

show_graphs = True
atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
ncm = ncm_options[0]
atlas = atlases[0]
regularization = reg_options[1]

nw = len(weights)
traits = ['CogTotalComp_AgeAdj', 'CogFluidComp_AgeAdj', 'CogCrystalComp_AgeAdj']
number_of_iterations = 50
knum = 8
columns = [str(k) for k in range(1, knum+1)]
explained_var_table = pd.DataFrame(index = weights, columns = columns)

model_r2_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_F_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_p_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_CVrmse_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_accuracy_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_results_dict = {'R2_adj': model_r2_table, 'F': model_F_table, 'p-value': model_p_table,'CV(RMSE)': model_CVrmse_table,'Accuracy': model_accuracy_table}
model_allruns = dict()
model_allruns_mean = dict()

for model_result in model_results_dict:
    model_allruns[model_result] = pd.DataFrame(columns=['value','trait','weight','iter'])
figs_folder = r'G:\data\V7\HCP\pca analysis results\Figs\Intelligence multiple matrices - Dist groups - multi runs'
all_var_table = pd.DataFrame(columns = ['component', 'explained_var','sub_network'])

coefficient_table = pd.DataFrame(columns = columns)
for trait_name in traits:
    dist_based_pca_dict = dict()
    n_components_per_dist_group_dict = dict()
    all_var_table_dict = dict()
    dist_coeff_dict = {w: coefficient_table for w in weights}
    for run in range(1, number_of_iterations+1):
        print('run: ', run)
        for weight_by in weights:
            all_var_table_dict[weight_by] = all_var_table
            if 'x' in weight_by:
                weight_by_couple = weight_by.split('x')
                dist_based_pca_dict[weight_by] = merge_dict_concatenate(dist_based_pca_dict[weight_by_couple[0]],
                                                                      dist_based_pca_dict[weight_by_couple[1]])
                n_components_per_dist_group_dict[weight_by] = merge_dict_sum(
                    n_components_per_dist_group_dict[weight_by_couple[0]],
                    n_components_per_dist_group_dict[weight_by_couple[1]])
                all_var_table_dict[weight_by] = pd.concat(
                    [all_var_table_dict[weight_by_couple[0]], all_var_table_dict[weight_by_couple[1]]])
                all_var_table_dict[weight_by] = all_var_table_dict[weight_by].sort_values('sub_network')
                dist_list = list(set(list(all_var_table_dict[weight_by]['sub_network'].sort_values())))

            else:
                subjects = glob.glob(
                    f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
                dist_mat = np.load(f'G:\data\V7\HCP\cm{os.sep}average_{atlas}_Dist_{regularization}_{ncm}.npy')

                connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
                dist_net_matrices, dist_net_mask_vec = from_whole_brain_to_dist_net(dist_mat, connectivity_matrices, knum)

                dist_based_pca_dict[weight_by], n_components_per_dist_group_dict[weight_by], explained_var_table, \
                all_var_table_dict[weight_by] = pca_for_each_network_different_number_of_components(dist_net_matrices,
                                                                                                    dist_net_mask_vec,
                                                                                                    explained_variance_th=0.2,
                                                                                                    explained_var_table=explained_var_table,
                                                                                                    weight_by=weight_by,
                                                                                                    all_var_table=
                                                                                                    all_var_table_dict[
                                                                                                        weight_by])
                dist_list = list(set(list(all_var_table_dict[weight_by]['sub_network'])))

            print(weight_by, regularization, ncm)
            print(f'*********** \n alpha: 1.0 \n ***********')

            trait_vector = load_trait_vector(trait_name, subjects)
            model, model_results_dict, mask_vec = linear_regression(dist_based_pca_dict[weight_by], trait_vector,
                                                                    dist_list,
                                                                    n_components=n_components_per_dist_group_dict[
                                                                        weight_by],
                                                                    model_results_dict=model_results_dict,
                                                                    weight_by=weight_by, trait_name=trait_name,
                                                                    lasso_regularized=True, alpha=1)
            print(f'Number of Predictors (after Lasso): {np.sum(model.params > 0)}')
            print(f'R2_adj: {model_results_dict["R2_adj"][trait_name][weight_by]}')
            print(f'p-value: {model_results_dict["p-value"][trait_name][weight_by]}')
            print(f'CV(RMSE): {100 * model_results_dict["CV(RMSE)"][trait_name][weight_by]}%')
            print(f'Accuracy: {model_results_dict["Accuracy"][trait_name][weight_by]}')
            print('***********')

            if model.f_pvalue < 0.05 and show_graphs:
                sum_table,color_dict = from_model_2_bar(model, dist_list, all_var_table_dict[weight_by],
                                             n_components_per_dist_group_dict[weight_by], trait_name, weight_by, ncm,
                                             regularization, atlas, figs_folder, show_all=False)
                dist_coeff_dict[weight_by] = dist_coeff_dict[weight_by].append(sum_table['percentage'], ignore_index=True)


    for result_value in model_results_dict:
        current_table = pd.DataFrame({'weight':list(model_results_dict[result_value].index),'trait':[trait_name]*nw,'iter':[run]*nw,'value':model_results_dict[result_value][trait_name].values})
        model_allruns[result_value] = model_allruns[result_value].append(current_table, ignore_index=True)


if show_graphs:
    for result_value in model_allruns:
        model_allruns_mean[result_value] = model_allruns[result_value].groupby(['weight', 'trait']).mean().reset_index()
        model_allruns_mean[result_value] = model_allruns_mean[result_value].pivot(index='weight', columns='trait')
        model_allruns_mean[result_value].columns = model_allruns_mean[result_value].columns.droplevel(0)
        model_allruns_mean[result_value] = model_allruns_mean[result_value].reindex(weights)
    label = f'{atlas}_{trait_name}_mean_{number_of_iterations}iters'
    model_heatmaps(model_allruns_mean, weights, figs_folder, label=label)


