import os, glob
from shutil import rmtree

hcp_folders = glob.glob(f'F:\data{os.sep}all_HCP\*{os.sep}')

for hf in hcp_folders:
    subjnum = str.split(hf, os.sep)[3]

    old_name = f'F:\data\V7\HCP{os.sep}{subjnum}{os.sep}'

    if os.path.exists(old_name):
        rmtree(hf)

