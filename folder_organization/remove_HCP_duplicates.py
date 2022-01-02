import os, glob
from shutil import rmtree

hcp_folders = glob.glob(f'F:\data{os.sep}all_HCP\*{os.sep}')

for hf in hcp_folders:
    subjnum = str.split(hf, os.sep)[3]

    old_name = f'F:\data\V7\HCP{os.sep}{subjnum}{os.sep}'

    if os.path.exists(old_name):
        rmtree(hf)



for hf in hcp_folders:
    f2del = glob.glob(f'{hf}data*')
    f2del.append(hf+'MPRAGE.nii')
    f2del.append(hf+'hifi_nodif_brain_mask.nii.gz')

    for f2d in f2del:
        if os.path.exists(f2d):
            os.remove(f2d)

sizes=[]
for sf in sfol:
    sf1 = os.path.join(sf,'data.nii.gz')
    try:
        f1size = os.path.getsize(sf1)
    except FileNotFoundError:
        sf1 = os.path.join(sf, 'data.nii')
        try:
            f1size = os.path.getsize(sf1)
        except FileNotFoundError:
            sizes.append(False)
            continue

    if f1size<1e7:
        sizes.append(False)
    else:
        sizes.append(True)


