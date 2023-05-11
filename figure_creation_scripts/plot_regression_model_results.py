import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def model_heatmaps(model_results_dict, weights, figs_folder = None, save_fig=True, traits = ['Total', 'Fluid', 'Crystal'], label=None):
    for i,trait in enumerate(traits):
        if '_' in trait:
            traits[i] = '_'.join(trait.split('_')[:-1])
    figure = plt.figure(figsize=(len(traits), 1+len(weights)/2))

    ax = sns.heatmap(model_results_dict['R2_adj'], linewidths=0.7, annot=True, fmt='.2f', cmap='mako', square=True,
                     xticklabels=traits, yticklabels=list(model_results_dict['R2_adj'].index))
    ax.set_title('Adjusted R2')
    if save_fig:
        plt.savefig(rf'{figs_folder}\Adjusted_R2_{label}.png')
    plt.show()

    figure = plt.figure(figsize=(len(traits), 1+len(weights)/2))

    ax = sns.heatmap(model_results_dict['CV(RMSE)'], linewidths=0.7, annot=True, fmt='.0%', cmap='mako', square=True,
                     xticklabels=traits, yticklabels=list(model_results_dict['CV(RMSE)'].index))
    ax.set_title('CV(RMSE)')
    if save_fig:
        plt.savefig(rf'{figs_folder}\RMSE_{label}.png')
    plt.show()

    # figure = plt.figure(figsize=(len(traits), 1+len(weights)/2))
    #
    # ax = sns.heatmap(model_results_dict['F'], linewidths=0.7, annot=True, fmt='.2f', cmap='mako', square=True,
    #                  xticklabels=traits, yticklabels=list(model_results_dict['F'].index))

    # figure = plt.figure(figsize=(len(traits), 1+len(weights)/2))
    #
    # ax = sns.heatmap(model_results_dict['Accuracy']/100, linewidths=0.7, annot=True, fmt='.0%', cmap='mako', square=True,
    #                  xticklabels=traits, yticklabels=list(model_results_dict['Accuracy'].index))
    # ax.set_title('Accuracy')
    # if save_fig:
    #     plt.savefig(rf'{figs_folder}\Accuracy_{label}.png')
    # plt.show()
    # ax.set_title('F')
    # if save_fig:
    #     plt.savefig(rf'{figs_folder}\F_{label}.png')
    # plt.show()
    #
    # figure = plt.figure(figsize=(len(traits), 1+len(weights)/2))
    #
    # ax = sns.heatmap(model_results_dict['p-value'], linewidths=0.7, annot=True, fmt='.0e', cmap='mako', vmax=0.05,
    #                  square=True, xticklabels=traits, yticklabels=list(model_results_dict['p-value'].index), norm=LogNorm())
    # ax.set_title('p-value')
    # if save_fig:
    #     plt.savefig(rf'{figs_folder}\p-value_{label}.png')
    # plt.show()

