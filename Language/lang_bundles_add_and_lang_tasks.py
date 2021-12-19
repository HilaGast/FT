from mean_axcaliber_bundle import mean_bundle_weight
import os, glob
from reading_from_xls.read_details_from_subject_table import *
from draw_scatter_fit import *

wos1 = []
lws = []
mean_bundle_AF_L=[]
mean_bundle_AF_R=[]
mean_bundle_SLF_L=[]
mean_bundle_SLF_R=[]

table1 = SubjTable(r'C:\Users\Admin\Desktop\Language\Subject list - Language.xlsx', 'Sheet1')
main_subj_folders = r'C:\Users\Admin\Desktop\Language'
bundles = ['AF_L','AF_R','SLF_L','SLF_R']

for sub in glob.glob(f'{main_subj_folders}{os.sep}*{os.sep}'):
    sn = sub.split(os.sep)[-2]
    folder_name = sub[:-1]
    L_bundle_file_name = f'{folder_name}{os.sep}streamlines{os.sep}_{bundles[0]}_mct01rt20_5d.trk'
    mean_bundle_AF_L.append(mean_bundle_weight(folder_name, L_bundle_file_name))

    R_bundle_file_name = f'{folder_name}{os.sep}streamlines{os.sep}_{bundles[1]}_mct01rt20_5d.trk'
    mean_bundle_AF_R.append(mean_bundle_weight(folder_name, R_bundle_file_name))


    L_bundle_file_name = f'{folder_name}{os.sep}streamlines{os.sep}_{bundles[2]}_mct01rt20_5d.trk'
    mean_bundle_SLF_L.append(mean_bundle_weight(folder_name, L_bundle_file_name))

    R_bundle_file_name = f'{folder_name}{os.sep}streamlines{os.sep}_{bundles[3]}_mct01rt20_5d.trk'
    mean_bundle_SLF_R.append(mean_bundle_weight(folder_name, R_bundle_file_name))

    wos1.append(table1.find_value_by_scan_Language('Word Order Score 1', sn))
    lws.append(table1.find_value_by_scan_Language('Learning words slope', sn))



draw_scatter_fit(wos1, mean_bundle_AF_L, comp_reg= True, ttl='wos1-AF_L')
draw_scatter_fit(lws, mean_bundle_AF_L, comp_reg=True, ttl='lws-AF_L')

draw_scatter_fit(wos1, mean_bundle_AF_R, comp_reg= True, ttl='wos1-AF_R')
draw_scatter_fit(lws, mean_bundle_AF_R, comp_reg=True, ttl='lws-AF_R')


draw_scatter_fit(wos1, mean_bundle_SLF_L, comp_reg= True, ttl='wos1-SLF_L')
draw_scatter_fit(lws, mean_bundle_SLF_L, comp_reg=True, ttl='lws-SLF_L')

draw_scatter_fit(wos1, mean_bundle_SLF_R, comp_reg= True, ttl='wos1-SLF_R')
draw_scatter_fit(lws, mean_bundle_SLF_R, comp_reg=True, ttl='lws-SLF_R')