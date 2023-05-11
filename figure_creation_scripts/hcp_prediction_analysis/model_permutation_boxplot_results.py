import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from HCP_network_analysis.hcp_cm_parameters import weights
import pandas as pd
import seaborn as sb


table = pd.read_excel(r'G:\data\V7\HCP\pca analysis results\Figs\xgboost_results - others\permutation_results_alpha1_test20_N1000_rand_20-30.xlsx')
table = table[[f'{weights[0]} Sub-networks', f'{weights[1]} Sub-networks', f'{weights[2]} Sub-networks',f'{weights[0]} Whole-brain', f'{weights[1]} Whole-brain', f'{weights[2]} Whole-brain']]

table.columns = ['Num \n \nSub-networks','FA \n \nSub-networks','ADD \n \nSub-networks','Num \n \nwhole-brain','FA \n \nwhole-brain','ADD \n \nwhole-brain']
sb.set_theme(
    rc={'figure.figsize': (30, 20), 'axes.grid': False,'axes.facecolor': 'white', 'xtick.labelsize': 36, 'ytick.labelsize': 40})
sb.boxplot(data=table,width=0.5, linewidth=5, palette=[[0.2, 0.7, 0.6],[0.3, 0.3, 0.5],[0.8, 0.5, 0.3],[0.2, 0.7, 0.6],[0.3, 0.3, 0.5],[0.8, 0.5, 0.3]])
plt.ylim(-0.1,0.4)
plt.show()


table = pd.read_excel(r'G:\data\V7\HCP\pca analysis results\Figs\xgboost_results - others\permutation_results_alpha1_test20_N1000_rand_all.xlsx',sheet_name='all')
sb.set_theme(
    rc={'figure.figsize': (30, 20), 'axes.grid': False,'axes.facecolor': 'white', 'xtick.labelsize': 36, 'ytick.labelsize': 40})
sb.boxplot(data=table,width=0.5, linewidth=5, palette=[[0.2, 0.7, 0.6],[0.3, 0.3, 0.5],[0.8, 0.5, 0.3],[0.2, 0.7, 0.6],[0.3, 0.3, 0.5],[0.8, 0.5, 0.3],[0.2, 0.7, 0.6],[0.3, 0.3, 0.5],[0.8, 0.5, 0.3],[0.2, 0.7, 0.6],[0.3, 0.3, 0.5],[0.8, 0.5, 0.3]])
plt.ylim(-0.1,0.4)
plt.show()

table = pd.read_excel(r'G:\data\V7\HCP\pca analysis results\Figs\xgboost_results - PCA20 - Leave_one_out\leave_one_out_table_Num_N100.xlsx')
sb.set_theme(
    rc={'figure.figsize': (30, 20), 'axes.grid': False,'axes.facecolor': 'white', 'xtick.labelsize': 36, 'ytick.labelsize': 40})
sb.boxplot(data=table,width=0.5, linewidth=5, palette=[[0.6,0.6,0.6],[0.2, 0.7, 0.6],[0.2, 0.7, 0.6],[0.2, 0.7, 0.6],[0.2, 0.7, 0.6],[0.2, 0.7, 0.6],[0.2, 0.7, 0.6],[0.2, 0.7, 0.6],[0.2, 0.7, 0.6],[0.2, 0.7, 0.6],[0.2, 0.7, 0.6]])
plt.ylim(-0.1,0.4)
plt.show()

table = pd.read_excel(r'G:\data\V7\HCP\pca analysis results\Figs\xgboost_results - PCA20 - Leave_one_out\leave_one_out_table_FA_N100.xlsx')
sb.set_theme(
    rc={'figure.figsize': (30, 20), 'axes.grid': False,'axes.facecolor': 'white', 'xtick.labelsize': 36, 'ytick.labelsize': 40})
sb.boxplot(data=table,width=0.5, linewidth=5, palette=[[0.6,0.6,0.6],[0.3, 0.3, 0.5],[0.3, 0.3, 0.5],[0.3, 0.3, 0.5],[0.3, 0.3, 0.5],[0.3, 0.3, 0.5],[0.3, 0.3, 0.5],[0.3, 0.3, 0.5],[0.3, 0.3, 0.5],[0.3, 0.3, 0.5],[0.3, 0.3, 0.5]])
plt.ylim(-0.1,0.4)
plt.show()

table = pd.read_excel(r'G:\data\V7\HCP\pca analysis results\Figs\xgboost_results - PCA20 - Leave_one_out\leave_one_out_table_ADD_N100.xlsx')
sb.set_theme(
    rc={'figure.figsize': (30, 20), 'axes.grid': False,'axes.facecolor': 'white', 'xtick.labelsize': 36, 'ytick.labelsize': 40})
sb.boxplot(data=table,width=0.5, linewidth=5, palette=[[0.6,0.6,0.6],[0.8, 0.5, 0.3],[0.8, 0.5, 0.3],[0.8, 0.5, 0.3],[0.8, 0.5, 0.3],[0.8, 0.5, 0.3],[0.8, 0.5, 0.3],[0.8, 0.5, 0.3],[0.8, 0.5, 0.3],[0.8, 0.5, 0.3],[0.8, 0.5, 0.3]])
plt.ylim(-0.1,0.4)
plt.show()


from scipy.stats import wilcoxon
x1 = table['Full']
x2 = table['SomMot']
x3 = table['Vis']
x4 = table['Cont']
x5 = table['Default']
x6 = table['SalVentAttn']
x7 = table['DorsAttn']
x8 = table['Limbic']
x9 = table['inter_network']
x10 = table['LH']
x11 = table['RH']

print(wilcoxon(x1,x2))
print(wilcoxon(x1,x3))
print(wilcoxon(x1,x4))
print(wilcoxon(x1,x5))
print(wilcoxon(x1,x6))
print(wilcoxon(x1,x7))
print(wilcoxon(x1,x8))
print(wilcoxon(x1,x9))
print(wilcoxon(x1,x10))
print(wilcoxon(x1,x11))
