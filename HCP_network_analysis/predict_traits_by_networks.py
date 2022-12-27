import numpy as np
import glob, os
import pandas as pd
from draw_scatter_fit import draw_scatter_fit

def create_all_subject_connectivity_matrices(subjects):

    connectivity_matrices = []
    for subject in subjects:
        connectivity_matrices.append(np.load(subject))
    connectivity_matrices = np.array(connectivity_matrices)
    connectivity_matrices = np.swapaxes(connectivity_matrices, 0, -1)

    return connectivity_matrices


def from_whole_brain_to_networks(connectivity_matrices, atlas_index_labels):

    labels_file = open(atlas_index_labels, 'r', errors='ignore')
    labels_name = labels_file.readlines()
    labels_file.close()
    labels_networks = find_network_names(labels_name)
    network_mask_dict = create_dict_of_networks_indices(labels_name, labels_networks)
    networks_matrices, network_mask_vecs = create_networks_matrices(connectivity_matrices, network_mask_dict)

    return networks_matrices, network_mask_vecs


def find_network_names(labels_name):
    label_networks = []
    for l in labels_name:
        label_parts = l.split('\t')
        label_networks.append(label_parts[1].split('_')[2])
    label_networks = list(set(label_networks))

    return label_networks


def create_dict_of_networks_indices(labels_name, label_networks):
    network_mask_dict = {}
    for network in label_networks:
        network_mask_dict[network] = []
    for l in labels_name:
        label_parts = l.split('\t')
        network_mask_dict[label_parts[1].split('_')[2]].append(int(label_parts[0])-1)
    return network_mask_dict


def create_networks_matrices(connectivity_matrices, network_mask_dict):
    networks_matrices = {}
    all_masks = []
    network_mask_vecs = {}

    for network in network_mask_dict.keys():
        network_mask = np.zeros(connectivity_matrices.shape, dtype = bool)
        for r in network_mask_dict[network]:
            for c in network_mask_dict[network]:
                network_mask[r, c, :] = True
        networks_matrices[network] = connectivity_matrices*network_mask
        network_mask_vecs[network] = network_mask[:,:,0].flatten()
        all_masks.append(network_mask)
    all_masks = np.asarray(all_masks)
    all_masks = np.sum(all_masks, axis = 0)
    not_mask = np.logical_not(all_masks)
    networks_matrices['inter_network'] = connectivity_matrices*not_mask
    network_mask_vecs['inter_network'] = not_mask[:,:,0].flatten()

    return networks_matrices, network_mask_vecs


def pca_for_each_network(networks_matrices, network_mask_vecs, n_components = 2, explained_var_table = None, weight_by = ''):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    networks_pca = {}
    print(f'Total explained variance for {n_components} components:')
    for network in networks_matrices:
        network_vector = np.zeros((networks_matrices[network].shape[-1], np.sum(network_mask_vecs[network])))
        for s in range(networks_matrices[network].shape[-1]):
            vec = networks_matrices[network][:, :, s].flatten()
            network_vector[s,:] = vec[network_mask_vecs[network]].reshape(1,-1)
        network_vector = StandardScaler().fit_transform(network_vector)
        pca = PCA(n_components=n_components)
        networks_pca[network] = pca.fit_transform(network_vector)
        #show_explained_variance(pca)
        print(f'{network}: {round(100*np.cumsum(pca.explained_variance_ratio_)[-1])}%')

        if type(explained_var_table) == pd.core.frame.DataFrame:
            explained_var_table[network][weight_by] = round(100*np.cumsum(pca.explained_variance_ratio_)[-1])

    return networks_pca, explained_var_table


def pca_for_each_network_different_number_of_components(networks_matrices, network_mask_vecs, explained_variance_th = 0.5, explained_var_table = None, weight_by = '', all_var_table = None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    networks_pca = {}
    n_components_per_network = {}
    for network in networks_matrices:
        network_vector = np.zeros((networks_matrices[network].shape[-1], np.sum(network_mask_vecs[network])))
        for s in range(networks_matrices[network].shape[-1]):
            vec = networks_matrices[network][:, :, s].flatten()
            network_vector[s,:] = vec[network_mask_vecs[network]].reshape(1,-1)
        network_vector = StandardScaler().fit_transform(network_vector)
        n_components = 2
        pca = PCA(n_components=n_components)
        n_pca = pca.fit_transform(network_vector)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)[-1]

        while explained_variance < explained_variance_th:
            n_components += 1
            pca = PCA(n_components=n_components)
            n_pca = pca.fit_transform(network_vector)
            explained_variance = np.cumsum(pca.explained_variance_ratio_)[-1]
        #print(f'Total explained variance for {n_components} components:')
        #print(f'{network}: {np.round(100 * explained_variance)}%')
        if all_var_table is not None:
            all_var_table = all_var_table.append(pd.DataFrame({'explained_var':pca.explained_variance_ratio_,'sub_network': [network]*n_components, 'component': list(range(1,n_components+1))}))

        networks_pca[network] = n_pca
        n_components_per_network[network] = n_components
        if type(explained_var_table) == pd.core.frame.DataFrame:
            explained_var_table[network][weight_by] = n_components

    return networks_pca, n_components_per_network, explained_var_table, all_var_table


def positive_negative_components_for_each_network(connectivity_matrices, networks_matrices, trait_vector, th = 0.005):

    positive_mask, negative_mask = create_positive_negative_mask(connectivity_matrices, trait_vector, th)
    network_positive_negative_components = {}
    for network in networks_matrices:
        positives = networks_matrices[network]*positive_mask
        positives[positives == 0] = np.nan
        negatives = networks_matrices[network]*negative_mask
        negatives[negatives == 0] = np.nan
        network_positive_negative_components[network] = [np.nanmean(positives, axis = (0, 1)), np.nanmean(negatives, axis = (0, 1))]

    return network_positive_negative_components


def create_positive_negative_mask(connectivity_matrices, trait_vector, th):
    from scipy.stats import pearsonr

    nan_mask = np.isnan(trait_vector)
    trait_vector = trait_vector[~nan_mask]
    positive_mask =  np.zeros(connectivity_matrices.shape, dtype = bool)
    negative_mask =  np.zeros(connectivity_matrices.shape, dtype = bool)

    for row in range(connectivity_matrices.shape[0]):
        for col in range(connectivity_matrices.shape[1]):
            x = connectivity_matrices[row, col, :]
            x = x[~nan_mask]
            r, p = pearsonr(x, trait_vector)
            if p < th and r > 0:
                positive_mask[row, col, :] = True
            elif p < th and r < 0:
                negative_mask[row, col, :] = True

    return positive_mask, negative_mask


def show_explained_variance(pca):
    import matplotlib.pyplot as plt
    cum_sum_eigenvalues = np.cumsum(pca.explained_variance_ratio_)
    plt.step(range(1, len(cum_sum_eigenvalues)+1), cum_sum_eigenvalues, where='mid')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


def load_trait_vector(trait_name, subjects):
    import pandas as pd

    table1 = pd.read_csv('G:\data\V7\HCP\HCP_behavioural_data.csv')
    trait_vector = []
    for sl in subjects:
        subj_id = sl.split('\\')[-3]
        trait_vector.append(float(table1[trait_name][table1['Subject']==int(subj_id)].values))
    trait_vector = np.asarray(trait_vector)


    return trait_vector


def remove_nans(network_components,trait_vector):
    network_components_nonan = network_components.copy()
    nan_mask = np.isnan(trait_vector)
    for network in network_components_nonan:
        network_components_nonan[network] = network_components_nonan[network][~nan_mask]
    trait_vector = trait_vector[~nan_mask]

    return network_components_nonan, trait_vector


def linear_regression(networks_components, trait_vector, network_list, n_components = 2, model_results_dict = None, weight_by = '', trait_name = '', lasso_regularized = False, alpha = 0.01):
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    from statsmodels.tools.eval_measures import rmse

    X, y = create_X_y(networks_components, trait_vector, network_list, n_components)
    mask_vec = []
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    if lasso_regularized:
        model = sm.OLS(y_train, X_train, 'drop').fit_regularized(alpha=alpha, L1_wt=1, refit=True) #Lasso
        mask_vec = model.params != 0

    else:
        model = sm.OLS(y_train, X_train,'drop').fit() #No regularization
    #print(model.summary())
    y_pred = model.predict(X_test)

    if type(model_results_dict) == dict:

        model_results_dict['R2_adj'][trait_name][weight_by] = model.rsquared_adj
        model_results_dict['F'][trait_name][weight_by] = model.fvalue
        model_results_dict['p-value'][trait_name][weight_by] = model.f_pvalue
        model_results_dict['CV(RMSE)'][trait_name][weight_by] = rmse(y_test, y_pred)/np.nanmean(y_test)
        errors = abs(y_pred - y_test)
        mape = 100 * (errors / y_test) # mean absolute percentage error
        accuracy = 100 - np.mean(mape)
        model_results_dict['Accuracy'][trait_name][weight_by] = accuracy # %

    return model, model_results_dict, mask_vec


def create_X_y(networks_components, trait_vector, network_list, n_components = 2):

    networks_components, trait_vector = remove_nans(networks_components, trait_vector)
    if type(n_components)==int:
        X = np.zeros((len(trait_vector), len(networks_components)*n_components))
        for i, network in enumerate(network_list):
            for n in range(n_components):
                X[:, i*n_components+n] = networks_components[network][:, n]
    elif type(n_components)==dict:
        X = np.zeros((len(trait_vector), sum(n_components.values())))
        current_i = 0
        for i, network in enumerate(network_list):
            for n in range(n_components[network]):
                X[:, current_i] = networks_components[network][:, n]
                current_i += 1
    y = trait_vector

    return X, y



if __name__ == '__main__':
    from HCP_network_analysis.hcp_cm_parameters import *

    atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
    ncm = ncm_options[0]
    atlas = atlases[0]
    weight_by = weights[2]
    regularization = reg_options[1]
    trait_name =  'CogTotalComp_AgeAdj'
    n_components = 2
    subjects = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
    connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
    networks_matrices, network_mask_vecs = from_whole_brain_to_networks(connectivity_matrices, atlas_index_labels)
    networks_pca = pca_for_each_network(networks_matrices, network_mask_vecs, n_components = n_components)
    #networks_pca, n_components_per_network = pca_for_each_network_different_number_of_components(networks_matrices, network_mask_vecs, explained_variance_th = 0.2)
    trait_vector = load_trait_vector(trait_name, subjects)
    print(weight_by, regularization, ncm)
    linear_regression(networks_pca, trait_vector, n_components = n_components)
