import pandas as pd


class SubjTable:

    def __init__(self, table_file_name='F:\data\MRI table update new.xlsx', sheet = 'THE BASE'):

        self.table = pd.read_excel(table_file_name,sheet)


    def find_value_by_scan(self, value_title, scan_name):

        value = self.table[value_title][self.table['SCAN FILE'] == scan_name].values

        return value

    def find_age_by_scan(self, scan_name):

        date_of_birth = self.find_value_by_scan('Date of Birth', scan_name)
        date_of_scan = self.find_value_by_scan('Date Of Scan', scan_name)
        age_days = date_of_scan-date_of_birth
        age = round(int(age_days[0].days)/365,1)

        return age
