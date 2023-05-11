from HCP_network_analysis.prediction_model.predict_traits_by_networks import *
from HCP_network_analysis.hcp_cm_parameters import *
from HCP_network_analysis.prediction_model.predict_traits_wb import whole_brain_matrix

show_graphs = True
atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
ncm = ncm_options[0]
atlas = atlases[0]
regularization = reg_options[0]

traits = ['CogTotalComp_AgeAdj']
explained_var_table = pd.DataFrame(index = weights[:3], columns = ['wb'])

figs_folder = r'G:\data\V7\HCP\pca analysis results\Figs\linear_regression\linear_regression_wb_PCA30'
for trait_name in traits:
    for weight_by in weights[:3]:
        all_var_table = pd.DataFrame(columns = ['component', 'explained_var','sub_network'])
        subjects = glob.glob(
            f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
        connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
        networks_matrices, network_mask_vecs = whole_brain_matrix(connectivity_matrices)
        networks_pca, n_components_per_network, explained_var_table, all_var_table = pca_for_each_network_different_number_of_components(networks_matrices,
                                                                                                     network_mask_vecs,
                                                                                                     explained_variance_th=0.3,explained_var_table=explained_var_table, weight_by=weight_by, all_var_table = all_var_table)
        trait_vector = load_trait_vector(trait_name, subjects)
        print(weight_by, regularization, ncm)
        print(f'*********** \n alpha: 1.0 \n ***********')
        network_list = list(set(list(all_var_table['sub_network'])))
        model, mask_vec, r, p = linear_regression(networks_pca, trait_vector, network_list,
                                                                    n_components=n_components_per_network,
                                                                    weight_by=weight_by, trait_name=trait_name,
                                                                    lasso_regularized=True, alpha=1, figs_folder=figs_folder, rs=0)





