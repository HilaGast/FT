import os, glob
from shutil import copy

ronnies_folders = glob.glob(f'H:\HCP\*{os.sep}')
#sf_name = [f'raw_data{os.sep}bvals',f'raw_data{os.sep}bvecs',f'raw_data{os.sep}brain_mask.nii.gz',f'cm{os.sep}HCP.npy',f'cm{os.sep}HCP_lookup.npy',f'cm{os.sep}HCP_unsorted.npy',f'cm{os.sep}HCP_unsorted.npy',f'cm{os.sep}HCP_lookup_unsorted.npy']
#df_name = ['data.bval','data.bvec','data_brain_mask.nii.gz','cm_num.npy','cm_num_lookup.npy','cm_num_unsorted.npy','cm_num_lookup_unsorted.npy']
sf_name = [f'raw_data{os.sep}mprage.nii.gz']
df_name = ['MPRAGE.nii']
for rf in ronnies_folders[::]:
    subjnum = str.split(rf, os.sep)[2]
    dir_name = f'F:\data{os.sep}all_HCP{os.sep}{subjnum}{os.sep}'
    org_name = f'H:\HCP{os.sep}{subjnum}{os.sep}'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    if not os.path.exists(f'{dir_name}streamlines'):
        os.mkdir(f'{dir_name}streamlines')

    for sfn,dfn in zip(sf_name,df_name):
        sf = f'{org_name}{sfn}'
        df = f'{dir_name}{dfn}'
        try:
            copy(sf,df)
        except FileNotFoundError:
            continue


#shortlist = glob.glob(f'F:\data\V7\HCP\*{os.sep}')
#for sf in shortlist:
#    subjnum = str.split(sf, os.sep)[-2]
#    dir_name = f'F:\data\V7\HCP{os.sep}{subjnum}{os.sep}'
#    org_name = f'H:\HCP{os.sep}{subjnum}{os.sep}'
#    if not os.path.exists(f'{dir_name}streamlines'):
#        os.mkdir(f'{dir_name}streamlines')
#    sf = f'{org_name}tracts\HCP.tck'
#    df = f'{dir_name}streamlines\HCP_tracts.tck'
#    copy(sf, df)
#
#    sf = f'{org_name}raw_data\mprage.nii.gz'
#    df = f'{dir_name}MPRAGE.nii'
#    copy(sf, df)


