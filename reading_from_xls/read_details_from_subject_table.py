import pandas as pd
import numpy as np

class SubjTable:

    def __init__(self, table_file_name='G:\data\MRI table update new.xlsx', sheet = 'THE BASE'):

        self.table = pd.read_excel(table_file_name,sheet)


    def find_value_by_scan(self, value_title, scan_name):

        value = self.table[value_title][self.table['SCAN FILE'] == scan_name].values

        return value

    def find_value_by_scan_Language(self, value_title, scan_name):

        value = self.table[value_title][self.table['Scan Name'] == scan_name].values
        if len(value)==0:
            value = self.table[value_title][self.table['Earlier scan name'] == scan_name].values
        if len(value) == 0:
            value = np.nan
        return float(value)

    def find_age_by_scan(self, scan_name):

        date_of_birth = self.find_value_by_scan('Date of Birth', scan_name)
        date_of_scan = self.find_value_by_scan('Date Of Scan', scan_name)
        age_days = date_of_scan-date_of_birth
        age = round(int(age_days[0].days)/365,1)

        return age

    def find_value_by_idx_balance(self, idx, value_title = 'balance improvement (ΔRMS)'):
        value = self.table[value_title][self.table['index'] == str(idx)].values

        if len(value) == 0:
            value = np.nan

        return float(value)