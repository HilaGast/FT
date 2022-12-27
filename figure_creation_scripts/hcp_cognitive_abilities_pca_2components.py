import numpy as np
import matplotlib.pyplot as plt

from HCP_network_analysis.predict_traits_by_networks import *
from HCP_network_analysis.hcp_cm_parameters import *

atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
ncm = ncm_options[0]
atlas = atlases[0]
regularization = reg_options[1]
n_components = 2


traits = ['CogTotalComp_AgeAdj', 'CogFluidComp_AgeAdj', 'CogCrystalComp_AgeAdj']
explained_var_table = pd.DataFrame(index = weights, columns = ['SomMot', 'Vis', 'Cont', 'Default', 'SalVentAttn', 'DorsAttn', 'Limbic', 'inter_network'])
#col_names = pd.MultiIndex.from_product([weights, ['R2_adj', 'F', 'p-value']], names = ['weight', 'statistic'])

model_r2_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_F_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_p_table = pd.DataFrame(index = weights, columns = traits, dtype='float')
model_results_dict = {'R2_adj': model_r2_table, 'F': model_F_table, 'p-value': model_p_table}

for trait_name in traits:
    for weight_by in weights:
        subjects = glob.glob(
            f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
        connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
        networks_matrices, network_mask_vecs = from_whole_brain_to_networks(connectivity_matrices, atlas_index_labels)
        networks_pca, explained_var_table = pca_for_each_network(networks_matrices, network_mask_vecs,
                                                                 n_components=n_components,
                                                                 explained_var_table=explained_var_table,
                                                                 weight_by=weight_by)
        trait_vector = load_trait_vector(trait_name, subjects)
        print(weight_by, regularization, ncm)
        model, model_results_dict = linear_regression(networks_pca, trait_vector, n_components=n_components,
                                                       model_results_dict=model_results_dict, weight_by=weight_by, trait_name=trait_name)
    # ax = explained_var_table.plot.bar(rot=0, figsize=(20, 15))
    # plt.title(f'Explained variance for {trait_name}', fontsize=25)
    # plt.ylabel('Explained variance [%]', fontsize=25)
    # plt.legend(fontsize=15)
    # plt.xticks(fontsize=25)
    # plt.yticks(fontsize=25)
    # plt.show()
import seaborn as sns

ax = sns.heatmap(model_results_dict['R2_adj'], linewidths=1, annot=True, fmt='.2f', cmap='mako',square=True, xticklabels=['Total', 'Fluid', 'Crystal'], yticklabels=weights)
ax.set_title('R2_adj')
plt.show()

ax = sns.heatmap(model_results_dict['F'], linewidths=1, annot=True, fmt='.2f', cmap='mako',square=True, xticklabels=['Total', 'Fluid', 'Crystal'], yticklabels=weights)
ax.set_title('F')
plt.show()

ax = sns.heatmap(model_results_dict['p-value'], linewidths=1, annot=True, fmt='.3f', cmap='mako',vmax=0.05,square=True, xticklabels=['Total', 'Fluid', 'Crystal'], yticklabels=weights)
ax.set_title('p-value')
plt.show()


