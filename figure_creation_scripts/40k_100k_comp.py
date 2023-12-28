import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from draw_scatter_fit import remove_nans
from figure_creation_scripts.weights_correlations import draw_scattter_fit_weights

num40 = np.load(r'G:\data\V7\HCP\cm\average_yeo7_200_Num_Org_SC_40k.npy')
num100 = np.load(r'G:\data\V7\HCP\cm\average_yeo7_200_Num_Org_SC_100k.npy')
add40 = np.load(r'G:\data\V7\HCP\cm\average_yeo7_200_ADD_Org_SC_40k.npy')
add100 = np.load(r'G:\data\V7\HCP\cm\average_yeo7_200_ADD_Org_SC_100k.npy')
fa40 = np.load(r'G:\data\V7\HCP\cm\average_yeo7_200_FA_Org_SC_40k.npy')
fa100 = np.load(r'G:\data\V7\HCP\cm\average_yeo7_200_FA_Org_SC_100k.npy')

num40 = np.triu(num40, 0)
num100 = np.triu(num100, 0)
add40 = np.triu(add40, 0)
add100 = np.triu(add100, 0)
fa40 = np.triu(fa40, 0)
fa100 = np.triu(fa100, 0)

num40[num40 == 0] = np.nan
num100[num100 == 0] = np.nan
add40[add40 == 0] = np.nan
add100[add100 == 0] = np.nan
fa40[fa40 == 0] = np.nan
fa100[fa100 == 0] = np.nan

draw_scattter_fit_weights(num40,num100, 'NOS - 40k', 'NOS - 100k', (1,70), (1,70))