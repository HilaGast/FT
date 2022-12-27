import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import t

def network_2_colors_dict(network_list):
    networks = list(set(network_list))
    networks.sort()
    colors = ['tomato', 'lightblue', 'lightgreen', 'gray', 'khaki', 'orange', 'plum', 'pink']
    colors_dict = dict(zip(networks, colors))
    return colors_dict

def create_network_col(network_list, all_var_table, n_components_per_network):

    n_components_per_network = redefine_n_components_per_network(all_var_table, network_list, n_components_per_network)
    network_col = ['constant']
    for network in network_list:
        network_col.extend([network]*n_components_per_network[network])

    return network_col

def redefine_n_components_per_network(all_var_table, network_list, n_components_per_network):
    for network in network_list:
        n_components_per_network[network] = len(all_var_table[all_var_table['sub_network'] == network])
    return n_components_per_network


def from_model_2_bar(model, network_list, all_var_table, n_components_per_network, trait_name, weight_by, ncm, regularization, atlas, figs_folder = None, show_all=True, ax=None, label=None, save_fig=True):
    if label is None:
        label = f'{weight_by}_{ncm}_{regularization}_{atlas}_{trait_name}'
    network_col = create_network_col(network_list, all_var_table, n_components_per_network)
    table = pd.DataFrame({'t-value': model.tvalues, 'p-value': model.pvalues,'coeff':model.params,'network':network_col})

    table = table[1:]

    table['is_significant'] = table['p-value'] < 0.05
    table['significant_coeff'] = table['is_significant'] * abs(table['coeff'])
    table = order_table_by_pvalue_for_each_network(table)

    color_dict = network_2_colors_dict(network_list)
    sum_table = pie_chart_of_coefficient_ratio_pca_components_from_each_network(table, color_dict, label, figs_folder)
    #pie_chart_of_significant_coefficient_pca_components_from_each_network(table, color_dict, label, figs_folder)
    if ax is None:
        ax = plt.gca()
    if show_all:
        t_th = abs(t.ppf(q=0.025, df = model.nobs - 1))
        sns.barplot(x=table.index.sort_values(), y=abs(table['t-value']), hue=table['network'], palette=color_dict, width=7, ax=ax)
        ax.axhline(y=t_th, color='black', linestyle='--', linewidth=4)
        ax.axhline(y=-1 * t_th, color='black', linestyle='--', linewidth=4)

        ax.set_title(f't-value for {label}', fontsize=80)
        ax.set_ylabel('abs(t-value)', fontsize=80)
        ax.set_xlabel('PCA components', fontsize=80)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', labelsize=70)
        plt.ylim(-5, 5)
        plt.xlim(-5, len(table) + 5)
        ax.figure.set_size_inches(45, 45)
        ax.legend(title='network', fontsize=55, title_fontsize=65, loc='lower right')
        if save_fig:
            plt.savefig(rf'{figs_folder}\barplot_of_t_value_for_{label}.png')
        plt.show()

    return sum_table, color_dict



def pie_chart_of_significant_coefficient_pca_components_from_each_network(table, color_dict, label, figs_folder, save_fig=True):
    ax = plt.gca()
    table['coeff'] = abs(table['coeff'])
    sum_table = table.groupby('network').sum()
    #sum_table['total_pca_components'] = [len(table[table['network'] == index]) for index in sum_table.index]
    #sum_table['percentage'] = 100 * sum_table['is_significant'] / sum_table['total_pca_components']
    sum_table['percentage'] = 100 * sum_table['significant_coeff'] / (sum_table['coeff'].sum())
    sum_table = sum_table.sort_values(by='percentage', ascending=False)
    sum_table = sum_table[sum_table['percentage'] > 0]
    plt.pie(sum_table['percentage'], labels=sum_table.index, wedgeprops=dict(width=0.5),colors=[color_dict[network] for network in sum_table.index])
    plt.title(f'{label} \n Only significant', fontsize = 10)
    if save_fig:
        plt.savefig(f'{figs_folder}\pie_chart_of_significant_coeff_pca_from_each_network_{label}.png')
    plt.show()

def pie_chart_of_coefficient_ratio_pca_components_from_each_network(table, color_dict, label, figs_folder, show = False, save_fig=True):
    ax = plt.gca()
    table['coeff'] = abs(table['coeff'])
    sum_table = table.groupby('network').sum()
    sum_table['percentage'] = 100 * sum_table['coeff'] / (sum_table['coeff'].sum())
    sum_table = sum_table.sort_values(by='percentage', ascending=False)
    #sum_table = sum_table[sum_table['percentage'] > 0]

    if show:
        plt.pie(sum_table['percentage'], labels=sum_table.index, wedgeprops=dict(width=0.5),colors=[color_dict[network] for network in sum_table.index])
        plt.title(label, fontsize = 10)
        if save_fig:
            plt.savefig(f'{figs_folder}\pie_chart_of_coeff_pca_from_each_network_{label}.png')
        plt.show()
    return sum_table


def plot_coeff_over_iterations(net_coeff_dict, num_of_iters, trait, figs_folder, color_dict, save_fig=True, boxplot= True, pieplot=True):
    for weight in net_coeff_dict.keys():
        label = f'{trait}_{weight}'

        if boxplot:
            df = pd.DataFrame(net_coeff_dict[weight])
            ax = plt.gca()
            df = df.melt(var_name='network', value_name='coeff')
            sns.boxplot(x='network', y='coeff', data=df, ax=ax, palette=color_dict)
            ax.set_title(f'% Coefficient over {num_of_iters} iterations for \n {label}', fontsize=45)
            ax.set_ylabel('% Coefficient', fontsize=50)
            ax.set_xlabel('Network', fontsize=50)
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, labelsize=40,
                            rotation=30)
            plt.tick_params(axis='y', which='both', labelsize=40)
            ax.figure.set_size_inches(25, 20)
            if save_fig:
                plt.savefig(rf'{figs_folder}\boxplot_coeff_over_iterations_for_{label}.png')
            plt.show()

        if pieplot:
            plt.pie(net_coeff_dict[weight].mean(), labels=net_coeff_dict[weight].columns, wedgeprops=dict(width=0.5),
                    colors=[color_dict[network] for network in net_coeff_dict[weight].columns])
            ax.set_title(f'% Coefficient over {num_of_iters} iterations for \n {label}', fontsize=45)
            if save_fig:
                plt.savefig(rf'{figs_folder}\pie_coeff_over_iterations_for_{label}.png')
            plt.show()





def order_table_by_pvalue_for_each_network(table):
    table = table.sort_values(by=['network', 'p-value'], ascending=[True, True])

    return table