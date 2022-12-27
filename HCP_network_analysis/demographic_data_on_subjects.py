import numpy as np
import pandas as pd
import glob,os
from cc_analysis.cc_boxplot import load_mat_cc_file

def count_subjects(subj_list):
    short_list=[]
    for sl in subj_list:
        dir_name = sl[:-1]
        #try:
        #    load_mat_cc_file(dir_name)
        #except FileNotFoundError:
        #    continue
        if os.path.exists(sl+'raverage_add_map.nii'):
            subj_number = sl.split(os.sep)[-2]
            short_list.append(subj_number)
    print(f"{len(short_list)} subjects in group")
    return short_list

def count_females(table1, short_list):
    num_f = 0
    for sl in short_list:
        if table1['Gender'][table1['Subject']==int(sl)].values == 'F':
            num_f+=1
    print(f"{num_f} females in group")

def age_info(table1, short_list):
    ages = []
    for sl in short_list:
        ages.append(int(table1['Age_in_Yrs'][table1['Subject']==int(sl)].values))

    print(f"Age range: {min(ages)}-{max(ages)}, Mean: {np.mean(ages)}")

if __name__ == "__main__":
    subj_list = glob.glob(f'G:\data\V7\HCP\*{os.sep}')
    table1 = pd.read_csv('G:\data\V7\HCP\HCP_demographic_data.csv')
    short_list = count_subjects(subj_list)
    count_females(table1, short_list)
    age_info(table1, short_list)

