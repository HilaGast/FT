import glob, os
import nibabel as nib

from Tractography.files_loading import load_ft
from Tractography.tractography_vis import show_tracts_simple

main_fol = 'F:\Hila\TDI\siemens'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}T*{os.sep}')
exp = 'D60d11'

for subj_fol in all_subj_fol:
    subj = subj_fol.split(os.sep)[-2]
    print(subj)
    tract_path = f'{subj_fol}{exp}{os.sep}streamlines{os.sep}wb_csd_fa.tck'
    nii_ref = nib.load(f'{subj_fol}{exp}{os.sep}diff_corrected_{exp}.nii')
    s_list = list(load_ft(tract_path,nii_ref))
    show_tracts_simple(s_list,folder_name=subj_fol,fig_type='QC', down_samp=1000)