import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from HCP_network_analysis.prediction_model.predict_traits_wb import whole_brain_matrix
from figure_creation_scripts.plot_for_predictors_values import from_model_2_bar_fi
from HCP_network_analysis.prediction_model.predict_traits_by_networks import *
from HCP_network_analysis.hcp_cm_parameters import *
from figure_creation_scripts.plot_regression_model_results import model_heatmaps
from useful_tools.merge_dict_with_common_keys import *
from HCP_network_analysis.prediction_model.xgboost_linear_model import gradient_boost_model, \
    gradient_boost_permutation_test

show_graphs = True
atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
ncm = ncm_options[0]
atlas = atlases[0]
regularization = reg_options[0]
networks =['SomMot_LH', 'Vis_LH', 'Cont_LH', 'Default_LH', 'SalVentAttn_LH', 'DorsAttn_LH', 'Limbic_LH','SomMot_RH', 'Vis_RH', 'Cont_RH', 'Default_RH', 'SalVentAttn_RH', 'DorsAttn_RH', 'Limbic_RH','SomMot', 'Vis', 'Cont', 'Default', 'SalVentAttn', 'DorsAttn', 'Limbic', 'inter_network_LH', 'inter_network_RH']
trait_name = 'CogTotalComp_AgeAdj'
nw = len(weights)
N = 1000
model_params = {}
model_params['test_size'] = 0.2
model_params['alpha'] = 1  # 0.5 Lasso regularization alpha
model_params['eta'] = 0.05
model_params['n_estimators'] = 200
figs_folder = r'G:\data\V7\HCP\pca analysis results\Figs\xgboost_results - others'

explained_var_table_sn = pd.DataFrame(index = weights, columns = networks)
explained_var_table_wb = pd.DataFrame(index = weights, columns = ['wb'])

all_var_table = pd.DataFrame(columns = ['component', 'explained_var','sub_network'])

r = pd.DataFrame(columns = [f'{weights[0]} Sub-networks', f'{weights[1]} Sub-networks', f'{weights[2]} Sub-networks',f'{weights[0]} Whole-brain', f'{weights[1]} Whole-brain', f'{weights[2]} Whole-brain'])

for model in ['Sub-networks', 'Whole-brain']:
    if model == 'Sub-networks':
        networks_pca_dict = dict()
        n_components_per_network_dict = dict()
        all_var_table_dict = dict()
        for weight_by in weights[:3]:
            print(trait_name, weight_by, regularization, ncm)
            all_var_table_dict[weight_by] = all_var_table
            subjects = glob.glob(
                f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
            connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
            networks_matrices, network_mask_vecs = from_whole_brain_to_networks(connectivity_matrices,
                                                                                atlas_index_labels, hemi_flag=True)
            networks_pca_dict[weight_by], n_components_per_network_dict[weight_by], explained_var_table_sn, \
                all_var_table_dict[weight_by] = pca_for_each_network_different_number_of_components(networks_matrices,
                                                                                                    network_mask_vecs,
                                                                                                    explained_variance_th=0.1,
                                                                                                    explained_var_table=explained_var_table_sn,
                                                                                                    weight_by=weight_by,
                                                                                                    all_var_table=
                                                                                                    all_var_table_dict[
                                                                                                        weight_by])
            network_list = list(set(list(all_var_table_dict[weight_by]['sub_network'])))
            trait_vector = load_trait_vector(trait_name, subjects)

            all_r = gradient_boost_permutation_test(networks_pca_dict[weight_by], trait_vector, network_list,
                                                    n_components_per_network_dict[weight_by], model_params, trait_name,
                                                    weight_by, figs_folder, n=N)
            r[f'{weight_by} {model}'] = all_r


    elif model == 'Whole-brain':
        networks_pca_dict = dict()
        n_components_per_network_dict = dict()
        all_var_table_dict = dict()
        for weight_by in weights[:3]:
            print(trait_name, weight_by, regularization, ncm)
            all_var_table_dict[weight_by] = all_var_table
            subjects = glob.glob(
                f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
            connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
            networks_matrices, network_mask_vecs = whole_brain_matrix(connectivity_matrices)

            networks_pca_dict[weight_by], n_components_per_network_dict[weight_by], explained_var_table_wb, \
                all_var_table_dict[weight_by] = pca_for_each_network_different_number_of_components(networks_matrices,
                                                                                                    network_mask_vecs,
                                                                                                    explained_variance_th=0.2,
                                                                                                    explained_var_table=explained_var_table_wb,
                                                                                                    weight_by=weight_by,
                                                                                                    all_var_table=
                                                                                                    all_var_table_dict[
                                                                                                        weight_by])
            network_list = list(set(list(all_var_table_dict[weight_by]['sub_network'])))
            trait_vector = load_trait_vector(trait_name, subjects)

            all_r = gradient_boost_permutation_test(networks_pca_dict[weight_by], trait_vector, network_list,
                                                    n_components_per_network_dict[weight_by], model_params, trait_name,
                                                    weight_by, figs_folder, n=N)
            r[f'{weight_by} {model}'] = all_r


ax = sb.boxplot(data = r) #, palette={'Num Sub-network':[0.2, 0.7, 0.6], 'FA Sub-network':[0.3, 0.3, 0.5], 'ADD Sub-network':[0.8, 0.5, 0.3], 'Num Whole-brain':[0.2, 0.7, 0.6], 'FA Whole-brain':[0.3, 0.3, 0.5], 'ADD Whole-brain':[0.8, 0.5, 0.3]})
plt.show()

r.to_excel(rf'G:\data\V7\HCP\pca analysis results\Figs\xgboost_results - others\permutation_results_alpha1_test20_N1000_rand_10-20.xlsx')





