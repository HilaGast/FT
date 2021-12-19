from reading_from_xls.read_details_from_subject_table import *
import glob, os
from network_analysis.global_network_properties import *

table1 = SubjTable(r'C:\Users\Admin\Desktop\Language\Subject list - Language.xlsx','Sheet1')


eff_num = []
eff_fa = []
eff_add = []
wos1=[]
wos2=[]
wos3=[]
lws=[]
lwd=[]

main_subj_folder = r'C:\Users\Admin\Desktop\Language'
for sub in glob.glob(f'{main_subj_folder}{os.sep}*{os.sep}'):
    sn = sub.split(os.sep)[-2]


    num_mat_name = sub + 'non-weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
    if os.path.exists(num_mat_name):
        num_mat = np.load(num_mat_name)
        eff_num.append(get_efficiency(cm=num_mat))

        fa_mat_name = sub + 'weighted_wholebrain_5d_labmask_yeo7_200_FA_nonnorm.npy'
        fa_mat = np.load(fa_mat_name)
        eff_fa.append(get_efficiency(cm=fa_mat))

        add_mat_name = sub + 'weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
        add_mat = np.load(add_mat_name)
        eff_add.append(get_efficiency(cm=add_mat))

        wos1.append(table1.find_value_by_scan_Language('Word Order Score 1',sn))
        wos2.append(table1.find_value_by_scan_Language('Word Order Score 2',sn))
        wos3.append(table1.find_value_by_scan_Language('Word Order Score 3',sn))
        lws.append(table1.find_value_by_scan_Language('Learning words slope',sn))
        lwd.append(table1.find_value_by_scan_Language('Learning words dist',sn))



from draw_scatter_fit import *
draw_scatter_fit(wos1,eff_add,ttl=f'ADD Eglob Vs. Word Order Score',comp_reg=True)
draw_scatter_fit(lws,eff_add,ttl=f'ADD Eglob Vs. Learning Words Slope',comp_reg=True)

draw_scatter_fit(wos1,eff_num,ttl=f'Num Eglob Vs. Word Order Score',comp_reg=True)
draw_scatter_fit(lws,eff_num,ttl=f'Num Eglob Vs. Learning Words Slope',comp_reg=True)

