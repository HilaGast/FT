import glob, os
import pandas as pd
import nibabel as nib
import numpy as np


def load_vol(subj_fol,vol_name, scan_name):
    '''
    :param subj_fol:
    :param vol_name:
    :return:
    '''

    try:
        vol_mat = nib.load(f'{subj_fol}{scan_name}_3_2_midcc_{vol_name}.nii.gz').get_fdata()
    except FileNotFoundError:
        vol_mat = nib.load(f'{subj_fol}{scan_name}_{vol_name}.nii.gz').get_fdata()

    return vol_mat


def load_cc_mask(subj_fol, scan_name):
    import scipy.io as sio
    '''

    :param subj_fol:
    :return:
    '''
    mask_file = sio.loadmat(f'{subj_fol}{scan_name}_CC_mask.mat')

    slice_num = mask_file['SliceNum'][0][0]-1
    cc_mask = mask_file['masks']

    return cc_mask, slice_num


def load_wm_mask(exp_fol):
    import nibabel as nib
    mask_file = nib.load(f'{exp_fol}mprage_reg_mixeltype.nii').get_fdata()
    wm_mask = mask_file==2

    return wm_mask


def compute_vol_from_mask(vol_mat, cc_mask, slice_num):
    '''

    :param vol_mat:
    :param cc_mask:
    :return:
    '''
    vol_mat = np.transpose(vol_mat,(2,1,0))
    vol_mat = np.flipud(vol_mat)
    vol_mat = np.fliplr(vol_mat)
    vol_slice = vol_mat[:,:,slice_num]
    vec_vol_masks = []
    for m in range(0, cc_mask.shape[2]):
        vol = cc_mask[:,:,m]*vol_slice
        mean_mask = np.nanmean(vol[vol>0])
        std_mask = np.nanstd(vol[vol>0])
        vol[vol>mean_mask+2*std_mask] = 0
        vol[vol < mean_mask - 2 * std_mask] = 0
        vec_vol_masks.append(np.nansum(vol)/np.nansum(vol>0))

    return vec_vol_masks


def compute_vol_from_wm_mask(wm_mask, vol_mat):
    masked_vol = wm_mask * vol_mat
    mean_val = np.nanmean(masked_vol[masked_vol>0])

    return mean_val


def compute_mask_sizes(cc_mask):
    '''
    :param cc_mask:
    :return: vec_mask_sizes
    '''

    vec_mask_sizes=[]
    for m in range(0,cc_mask.shape[2]):
        vec_mask_sizes.append(np.nansum(cc_mask[:,:,m]))

    return vec_mask_sizes


if __name__ == '__main__':
    main_fol = 'Y:\qnap\siemens'
    all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}AxCaliber{os.sep}')
    all_subj_names = [s.split(os.sep)[3] for s in all_subj_fol]
    #vol_names = ['FA', 'MD', 'ADD', 'pFr','pH','pCSF'] #WB
    vol_names = ['CC vol','FA', 'MD', 'ADD', 'pFr','pH','pCSF'] #CC
    experiments = ['D60d11', 'D45d13', 'D31d18']
    parts = ["Total","G","AB","MB","PB","Sp"] #CC
    #multi_index = pd.MultiIndex.from_product([experiments,vol_names], names=["Experiment","Volume"]) #WB
    #table = pd.DataFrame(columns=multi_index, index=all_subj_names) #WB
    for experiment in experiments:
        multi_index = pd.MultiIndex.from_product([vol_names, parts], names=["Volume","Parts"]) #CC
        table = pd.DataFrame(columns=multi_index, index=all_subj_names) #CC
        scan_name = f"diff_corrected_{experiment}"
        for subj_fol,subj_name in zip(all_subj_fol, all_subj_names):
    #         exp_fol = f'{main_fol}{os.sep}{subj_name}{os.sep}{experiment}{os.sep}' #WB
    #         wm_mask = load_wm_mask(exp_fol) #WB
    #         for vol_name in vol_names:  # WB
    #             vol_mat = load_vol(subj_fol, vol_name, scan_name)#WB
    #             mean_vol_masked = compute_vol_from_wm_mask(wm_mask, vol_mat) #WB
    #             table.loc[subj_name,(experiment,vol_name)] = mean_vol_masked #WB
    # table.to_excel(f'Y:\qnap\siemens_results_WB{os.sep}results.xlsx') #WB


            cc_mask, slice_num = load_cc_mask(subj_fol,scan_name) #CC
            vec_mask_sizes = compute_mask_sizes(cc_mask) #CC
            table.loc[subj_name,"CC vol"] = vec_mask_sizes #CC
            for vol_name in vol_names[1:]: #CC
                 vol_mat = load_vol(subj_fol, vol_name, scan_name)#CC
                 vec_vol_mask = compute_vol_from_mask(vol_mat, cc_mask, slice_num) #CC
                 table.loc[subj_name, vol_name] = vec_vol_mask #CC
        table.to_excel(f'Y:\qnap\siemens_results_CC{os.sep}results_{scan_name.split("_")[2]}_new.xlsx') #CC

