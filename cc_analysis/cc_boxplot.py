import numpy as np
import pandas as pd
import os


def choose_group(group):
    subj = list()

    if group == 'hcp':
        subj_folder = r'F:\Hila\Ax3D_Pack\V6\v7calibration\hcp'
        names = [ name for name in os.listdir(subj_folder) if os.path.isdir(os.path.join(subj_folder, name)) ]
        for n in names:
            s = os.path.join(subj_folder,n)
            subj.append(s)
    elif group == 'thebase':
        subj_folder = r'F:\Hila\Ax3D_Pack\V6\v7calibration\TheBase4Ever'
        names = [ name for name in os.listdir(subj_folder) if os.path.isdir(os.path.join(subj_folder, name)) ]
        for n in names:
            s = os.path.join(subj_folder,n)
            subj.append(s)

    return subj_folder, subj


def load_mat_cc_file(folder_name):
    import scipy.io as sio
    cc_file = os.path.join(folder_name,'CC_mask.mat')
    mat_content = sio.loadmat(cc_file)
    slice_num = mat_content['SliceNum'][0][0]
    mask_genu = mat_content['masks'][:,:,1]
    mask_abody = mat_content['masks'][:,:,2]
    mask_mbody = mat_content['masks'][:,:,3]
    mask_pbody = mat_content['masks'][:,:,4]
    mask_splenium = mat_content['masks'][:,:,5]

    return slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium


def find_slice(folder_name, volume_name,slice_num, protocol):
    import nibabel as nib
    for file in os.listdir(folder_name):
        if volume_name in file and not file.startswith("r"):
            volume_file = os.path.join(folder_name,file)
            continue
    vol_img = nib.load(volume_file)
    vol_mat = vol_img.get_fdata()
    vol_mat = np.transpose(vol_mat,(2,1,0))
    vol_mat = np.flipud(vol_mat)
    vol_mat = np.fliplr(vol_mat)
    if protocol == 'hcp':
        vol_mat = np.flip(vol_mat,2)
    slice_vol = vol_mat[:,:,slice_num]

    return slice_vol


def calc_parameter_by_mask(parameter_type, slice_vol, mask):
    if  parameter_type == 'mean':
        parameter_val = np.nanmean(slice_vol[mask>0])
    elif parameter_type == 'median':
        parameter_val = np.nanmedian(slice_vol[mask > 0])
    else:
        print('Unrecognized parameter for calculation')

    return parameter_val


def create_cc_vioplot(cc_parts_table):
    import seaborn as sb
    import matplotlib.pyplot as plt
    ax = sb.violinplot(data = cc_parts_table, palette="Set1")
    plt.show()


def create_comperative_cc_vioplot(cc_parts_table, vio_type):
    import seaborn as sb
    import matplotlib.pyplot as plt
    if vio_type == 'split':
        ax = sb.violinplot(x = 'CC part', y= 'ADD', hue = 'Protocol', data = cc_parts_table, palette="Set2",split=True)
        plt.show()

    else:
        ax = sb.violinplot(x = 'CC part', y= 'ADD', hue = 'Protocol', data = cc_parts_table, palette="Set2")
        plt.show()


def show_vioplot_single_protocol(protocol):
    subj_folder, subj = choose_group(protocol)
    val_mat = np.zeros((len(subj),5))
    for i,folder_name in enumerate(subj):
        val_vec = []
        slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_mat_cc_file(folder_name)
        slice_vol = find_slice(folder_name, '3_2_AxPasi7', slice_num, protocol)
        for mask in [mask_genu,mask_abody,mask_mbody,mask_pbody,mask_splenium]:
            parameter_val = calc_parameter_by_mask('mean',slice_vol,mask)
            val_vec.append(parameter_val)
        val_mat[i,:] = val_vec

    cc_parts_table = pd.DataFrame(val_mat, columns = ['Genu','Anterior Body','Mid Body','Posterior Body','Splenium'])
    print(cc_parts_table)
    create_cc_vioplot(cc_parts_table)


def show_vioplot_compare_protocols(vio_type):
    subj_hcp = choose_group('hcp')[1]
    subj_thebase = choose_group('thebase')[1]

    protocol_list = ['HCP']*len(subj_hcp)*5+['The Base']*len(subj_thebase)*5
    parts_list = ['Genu','Anterior Body', 'Mid Body', 'Posterior Body', 'Splenium']*(len(subj_hcp)+len(subj_thebase))
    val_list = []
    for i,folder_name in enumerate(subj_hcp):
        slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_mat_cc_file(folder_name)
        slice_vol = find_slice(folder_name, '3_2_AxPasi7', slice_num,'hcp')
        for mask in [mask_genu,mask_abody,mask_mbody,mask_pbody,mask_splenium]:
            parameter_val = calc_parameter_by_mask('mean',slice_vol,mask)
            val_list.append(parameter_val)


    for i,folder_name in enumerate(subj_thebase):
        slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_mat_cc_file(folder_name)
        slice_vol = find_slice(folder_name, '3_2_AxPasi7', slice_num,'thebase')
        for mask in [mask_genu,mask_abody,mask_mbody,mask_pbody,mask_splenium]:
            parameter_val = calc_parameter_by_mask('mean',slice_vol,mask)
            val_list.append(parameter_val)

    d_vals = {'Protocol':protocol_list, 'CC part': parts_list,'ADD':val_list}
    cc_parts_table = pd.DataFrame(d_vals)
    create_comperative_cc_vioplot(cc_parts_table, vio_type)


if __name__ == '__main__':
    #show_vioplot_single_protocol('hcp')
    show_vioplot_compare_protocols(vio_type = 'sidebyside')
    #show_vioplot_compare_protocols(vio_type='split')


