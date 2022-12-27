def idx_2_label(atlas_labels_file, idx, atlas='bna', print_flag = False):

    labels_file = open(atlas_labels_file, 'r', errors='ignore')
    labels_name = labels_file.readlines()
    labels_file.close()

    if atlas == 'bna':
        labels_headers = read_labels_bna(labels_name, idx)
    if atlas == 'yeo7_200':
        labels_headers = read_labels_yeo7(labels_name, idx)
    if print_flag:
        for k,v in labels_headers.items():
            print(f'{str(k)} : {v}')

    return labels_headers


def read_labels_bna(labels_name, idx):

    labels_headers = {}

    for l in labels_name[1:]:
        label_parts = l.split('\t')
        if len(label_parts) == 5 and int(label_parts[2]) in idx:
            labels_headers[int(label_parts[2])] = f"{label_parts[1]} {' '.join(label_parts[4].split()[1:])}"

    return labels_headers

def read_labels_yeo7(labels_name, idx):

    labels_headers = {}

    for l in labels_name:
        label_parts = l.split('\t')
        if int(label_parts[0]) in idx:

            labels_headers[int(label_parts[0])] = ' '.join(label_parts[1].split('_')[1:])

    return labels_headers