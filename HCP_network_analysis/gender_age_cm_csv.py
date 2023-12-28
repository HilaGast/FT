import numpy as np
import pandas as pd
import glob,os

def return_gender(table, subject_name):
    return table['Gender'][table['Subject']==int(subject_name)].values[0]


def return_age(table, subject_name):
    return table['Age_in_Yrs'][table['Subject']==int(subject_name)].values[0]


if __name__ == "__main__":
    main_path = 'G:\data\V7\HCP'
    subj_list = glob.glob(f'{main_path}\*[0-9]{os.sep}')
    atlas = 'yeo7_100'
    cm_type = 'Num'
    demographic_table = pd.read_csv(f'{main_path}\HCP_demographic_data.csv')
    relevant_subj = []
    for subj in subj_list:
        if os.path.exists(subj + 'cm' + os.sep + f'{atlas}_{cm_type}_Org_SC_cm_ord.npy'):
            relevant_subj.append(subj)
    new_table = pd.DataFrame(index = range(len(relevant_subj)), columns=['CM','GENDER','AGE'])

    for i,subj in enumerate(relevant_subj):
        cm = np.load(subj + 'cm' + os.sep + f'{atlas}_{cm_type}_Org_SC_cm_ord.npy')
        subj_number = subj.split(os.sep)[-2]
        gender = return_gender(demographic_table, subj_number)
        age = return_age(demographic_table, subj_number)
        new_table['CM'][i] = cm
        new_table['GENDER'][i] = gender
        new_table['AGE'][i] = age

    new_table.to_pickle(f'{main_path}\{atlas}_{cm_type}cm_and_demographic_data.pkl')

    table = pd.read_pickle(f'{main_path}\{atlas}_{cm_type}cm_and_demographic_data.pkl')
