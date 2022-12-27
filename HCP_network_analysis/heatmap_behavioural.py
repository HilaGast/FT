import os, glob
import pandas as pd
from calc_corr_statistics.pearson_r_calc import *
from figure_creation_scripts.heatmap_tools import *

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')

table = pd.read_csv('G:\data\V7\HCP\HCP_behavioural_data.csv')


fields = ['Motor', 'Cognitive & Language', 'Sensory','All']
field = fields[0]

if field == 'Motor':
    cols = ['Endurance_AgeAdj', 'GaitSpeed_Comp', 'Dexterity_AgeAdj', 'Strength_AgeAdj']
    col_names = ['Endurance', 'GaitSpeed_Comp', 'Dexterity', 'Strength']
elif field == 'Cognitive & Language':
    cols = ['PicSeq_AgeAdj', 'CardSort_AgeAdj', 'Flanker_AgeAdj', 'PMAT24_A_CR', 'ListSort_AgeAdj',
            'CogTotalComp_AgeAdj', 'CogFluidComp_AgeAdj', 'CogCrystalComp_AgeAdj', 'ReadEng_AgeAdj', 'PicVocab_AgeAdj',
            'ProcSpeed_AgeAdj', 'IWRD_TOT', 'IWRD_RTC', 'Language_Task_Acc', 'Language_Task_Story_Acc']
    col_names = ['PicSeq', 'CardSort', 'Flanker', 'PMAT24', 'ListSort', 'CogTotal', 'CogFluid', 'CogCrystal', 'ReadEng',
                 'PicVocab', 'ProcSpeed', 'IWRD_TOT', 'IWRD_RTC', 'Language_Task', 'Task_Story']
elif field == 'Sensory':
    cols = ['Noise_Comp','Odor_AgeAdj','PainIntens_RawScore','Taste_AgeAdj','Mars_Final']
    col_names = ['Noise','Odor','PainIntens','Taste','Contrast']
elif field == 'All':
    cols = ['Endurance_AgeAdj', 'GaitSpeed_Comp', 'Dexterity_AgeAdj', 'Strength_AgeAdj','PicSeq_AgeAdj', 'CardSort_AgeAdj',
            'Flanker_AgeAdj', 'PMAT24_A_CR', 'ListSort_AgeAdj','CogTotalComp_AgeAdj',
            'CogFluidComp_AgeAdj', 'CogCrystalComp_AgeAdj', 'ReadEng_AgeAdj', 'PicVocab_AgeAdj',
            'ProcSpeed_AgeAdj', 'IWRD_TOT', 'IWRD_RTC', 'Language_Task_Acc', 'Language_Task_Story_Acc',
            'Noise_Comp','Odor_AgeAdj','PainIntens_RawScore','Taste_AgeAdj','Mars_Final']
    col_names = ['Endurance', 'GaitSpeed_Comp', 'Dexterity', 'Strength','PicSeq', 'CardSort', 'Flanker',
                 'PMAT24', 'ListSort', 'CogTotal', 'CogFluid', 'CogCrystal', 'ReadEng',
                 'PicVocab', 'ProcSpeed', 'IWRD_TOT', 'IWRD_RTC', 'Language_Task', 'Task_Story','Noise','Odor',
                 'PainIntens','Taste','Contrast']


table_r = pd.DataFrame(columns=cols, index= cols)
table_p = pd.DataFrame(columns=cols, index= cols)

for col_x in cols:
    for col_y in cols:
        vec_x = []
        vec_y = []
        for sl in subj_list:
            snum = str.split(sl, os.sep)[-2]
            vec_x.append(float(table[col_x][table['Subject'] == int(snum)].values))
            vec_y.append(float(table[col_y][table['Subject'] == int(snum)].values))

        r,p = calc_corr(vec_x, vec_y)
        table_r[col_x][col_y] = r
        table_p[col_x][col_y] = p

data = np.asarray(table_r,dtype='float')



fig, ax = plt.subplots()

im, cbar = heatmap(data, col_names, col_names, cbar_label='Pearson r')
texts = annotate_heatmap(im,textcolors=('white', 'white'),valfmt="{x:.1f}",fontsize=7)
fig.tight_layout()
plt.title(field)
plt.show()