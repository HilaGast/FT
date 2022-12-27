import matplotlib.pyplot as plt

def plot_components_after_lasso(all_var_table, trait_name, weight_by, mask_vec, figs_folder = None, save_fig = True):
    all_var_table = all_var_table[mask_vec[1:]]
    all_var_table['count'] = 1
    sum_table = all_var_table.groupby('sub_network').sum()
    title1 = f'After Lasso Regularization: \n {weight_by} {trait_name}'
    ax1, ax2 = sum_table.plot.bar(y=['count', 'explained_var'], rot=0, subplots=True, figsize=(10, 8), title=title1, colormap='Set2')
    ax1.set_ylim(0, 25)
    ax2.set_ylim(0, 0.25)
    if save_fig:
        plt.savefig(rf'{figs_folder}\Number_PCA_components_after_lasso_{weight_by}_{trait_name}.png')
    plt.show()

def plot_components_before_lasso(explained_var_table, trait_name, figs_folder = None, save_fig = True):
    ax = explained_var_table.plot.bar(rot=0, figsize=(20, 15),colormap='Set2')
    plt.title(f'Original number of PCA components for {trait_name}', fontsize=25)
    plt.ylabel('Number of PCA components', fontsize=25)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    if save_fig:
        plt.savefig(rf'{figs_folder}\Number_PCA_components_before_lasso_{trait_name}.png')
    plt.show()
