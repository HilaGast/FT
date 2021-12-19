import os
import pandas as pd


def network_id_list(main_atlas_folder = r'C:\Users\Admin\my_scripts\aal\yeo', atlas_type = 'yeo7_200', network_type = None, side = None):
    labels_file_name  = os.path.join(main_atlas_folder,atlas_type,'index2label.txt')

    labels_file = open(labels_file_name, 'r', errors='ignore')
    labels_name = labels_file.readlines()
    labels_file.close()
    labels_table = pd.DataFrame(columns=['label_num','hemi', 'network'],index=range(labels_name.__len__()))
    for i,line in enumerate(labels_name):
        if not line[0] == '#':
            line_parts = line.split()
            label_parts = line_parts[1].split('_')
            labels_table['label_num'][i] = int(line_parts[0])
            labels_table['hemi'][i] = str.lower(label_parts[1])
            labels_table['network'][i] = str.lower(label_parts[2])


    if network_type:
        id_net = list(labels_table['label_num'][labels_table['network'] == str.lower(network_type)])
    else:
        id_net = list(labels_table['label_num'])


    if side:
        id_side = list(labels_table['label_num'][labels_table['hemi'] == str.lower(side)])
    else:
        id_side = list(labels_table['label_num'])


    network_id = id_side and id_net

    return  network_id


