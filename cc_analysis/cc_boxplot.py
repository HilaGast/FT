import numpy as np
import pandas as pd
import os


def choose_group(group):
    subj = list()

    if group == 'hcp':
        subj_folder = r'G:\data\V7\hcp'
        names = [ name for name in os.listdir(subj_folder) if os.path.isdir(os.path.join(subj_folder, name)) ]
        for n in names:
            s = os.path.join(subj_folder,n)
            subj.append(s)
    elif group == 'thebase4ever':
        subj_folder = r'G:\data\V7\TheBase4Ever'
        names = [ name for name in os.listdir(subj_folder) if os.path.isdir(os.path.join(subj_folder, name)) ]
        for n in names:
            s = os.path.join(subj_folder,n)
            subj.append(s)

    elif group == 'thebase':
        subj_folder = r'G:\data\V7\TheBase'
        names = [ name for name in os.listdir(subj_folder) if os.path.isdir(os.path.join(subj_folder, name)) ]
        for n in names:
            s = os.path.join(subj_folder,n)
            subj.append(s)

    return subj_folder, subj


def choose_condition(main_path):
    subjD31 = list()
    subjD45 = list()
    subjD60 = list()

    names = [name for name in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, name))]
    for n in names:
        s = os.path.join(main_path, n)
        subjD31.append(os.path.join(s,'d18D31g7.19'))
        subjD45.append(os.path.join(s,'d13.2D45g7.69'))
        subjD60.append(os.path.join(s,'d11.3D60g7.64'))

    return subjD31, subjD45, subjD60



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
    sb.set_theme(
        rc={'figure.figsize': (14, 10), 'axes.facecolor': 'black', 'figure.facecolor': 'black', 'text.color': 'white',
            'xtick.color': 'white', 'ytick.color': 'white', 'axes.grid': False, 'grid.color': 'gray',
            'axes.labelcolor': 'white', 'axes.labelsize': 40, 'axes.labelweight': 'bold', 'legend.fontsize': 32,
            'legend.title_fontsize': 32, 'xtick.labelsize': 32, 'ytick.labelsize': 38})
    ax = sb.violinplot(data = cc_parts_table, color=[0.4, 0.7, 0.4],showmedians=True, linewidth=5) #), palette="set1")
    plt.show()

def create_cc_boxplot(cc_parts_table):
    import seaborn as sb
    import matplotlib.pyplot as plt
    ax = sb.boxplot(data = cc_parts_table, palette="Set1")
    plt.show()


def create_comperative_cc_vioplot(cc_parts_table, vio_type, split_by = 'Protocol'):
    import seaborn as sb
    import matplotlib.pyplot as plt
    if vio_type == 'split':
        sb.set_theme(rc={'figure.figsize':(14,18),'axes.facecolor':'black','figure.facecolor':'black', 'text.color': 'white', 'xtick.color': 'white', 'ytick.color': 'white','axes.grid':False,'grid.color':'gray','axes.labelcolor':'white','axes.labelsize':40,'axes.labelweight':'bold','legend.fontsize':32,'legend.title_fontsize':32,'xtick.labelsize':32,'ytick.labelsize':38})
        ax = sb.violinplot(x = 'CC_Part', y= 'ADD', hue = split_by, data = cc_parts_table,palette="Set2", split=True)
        plt.show()

    else:
        for part in set(cc_parts_table['CC_Part']):
            print(f'{part}:')
            print(np.nanmean(cc_parts_table["ADD [\u03BCm]"][cc_parts_table["CC_Part"] == part]))

        sb.set_theme(rc={'figure.figsize':(22,24),'axes.facecolor':'black','figure.facecolor':'black', 'text.color': 'white', 'xtick.color': 'white', 'ytick.color': 'white','axes.grid':False,'grid.color':'gray','axes.labelcolor':'white','axes.labelsize':50,'axes.labelweight':'bold','legend.fontsize':32,'legend.title_fontsize':34,'xtick.labelsize':34,'ytick.labelsize':38})
        ax = sb.violinplot(x = 'CC_Part', y= 'ADD [\u03BCm]', hue = split_by,scale='width', data = cc_parts_table,palette="Set2",showmedians=True, linewidth=5)

        ax.set_ylabel('eADD')
        ax.set_xlabel('\n CC parts')

        plt.show()


def create_comperative_cc_boxplot(cc_parts_table):
    import seaborn as sb
    import matplotlib.pyplot as plt
    sb.set_theme(
        rc={'figure.figsize': (18, 24), 'axes.facecolor': 'black', 'figure.facecolor': 'black', 'text.color': 'white',
            'xtick.color': 'white', 'ytick.color': 'white', 'axes.grid': False, 'grid.color': 'gray',
            'axes.labelcolor': 'white', 'axes.labelsize': 40, 'axes.labelweight': 'bold', 'legend.fontsize': 32,
            'legend.title_fontsize': 32, 'xtick.labelsize': 32, 'ytick.labelsize': 38})
    ax = sb.boxplot(x='CC_Part', y='ADD [\u03BCm]', hue='Protocol', data=cc_parts_table, palette="Set2")
    ax.set_ylim(4, 11)
    ax.set_ylabel('eADD')
    ax.set_xlabel('\n CC parts')
    plt.show()


def show_vioplot_single_protocol(protocol):
    subj_folder, subj = choose_group(protocol)
    #val_mat = np.zeros((len(subj),5))
    val_list = []
    for i,folder_name in enumerate(subj):
        val_vec = []
        try:
            slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_mat_cc_file(folder_name)
        except FileNotFoundError:
            continue
        slice_vol = find_slice(folder_name, '3_2_AxPasi7', slice_num, protocol)
        for mask in [mask_genu,mask_abody,mask_mbody,mask_pbody,mask_splenium]:
            parameter_val = calc_parameter_by_mask('median',slice_vol,mask)
            val_vec.append(parameter_val)
        val_list.append(val_vec)
    val_mat = np.reshape(val_list,(-1,5))

    cc_parts_table = pd.DataFrame(val_mat, columns = ['Genu','Anterior Body','Mid Body','Posterior Body','Splenium'])
    cc_parts_table,num_lo = detect_and_remove_outliers(cc_parts_table)
    print(cc_parts_table)
    print(f'Removed {num_lo} outliers')
    create_cc_vioplot(cc_parts_table)
    anova_for_cc_parts(cc_parts_table)
    #create_cc_boxplot(cc_parts_table)


def show_vioplot_compare_protocols(vio_type):
    subj_hcp = choose_group('hcp')[1]
    subj_thebase = choose_group('thebase')[1]
    subj_thebase4ever = choose_group('thebase4ever')[1]
    nhcp=0
    nthebase=0
    nthebase4ever=0
    val_list = []
    for i,folder_name in enumerate(subj_hcp):
        try:
            slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_mat_cc_file(folder_name)
        except FileNotFoundError:
            continue
        nhcp+=1
        slice_vol = find_slice(folder_name, '3_2_AxPasi7', slice_num,'hcp')
        for mask in [mask_genu,mask_abody,mask_mbody,mask_pbody,mask_splenium]:
            parameter_val = calc_parameter_by_mask('median',slice_vol,mask)
            val_list.append(parameter_val)


    for i,folder_name in enumerate(subj_thebase):
        try:
            slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_mat_cc_file(folder_name)
        except FileNotFoundError:
            continue
        nthebase+=1
        slice_vol = find_slice(folder_name, '3_2_AxPasi7', slice_num,'thebase')
        for mask in [mask_genu,mask_abody,mask_mbody,mask_pbody,mask_splenium]:
            parameter_val = calc_parameter_by_mask('median',slice_vol,mask)
            val_list.append(parameter_val)

    for i,folder_name in enumerate(subj_thebase4ever):
        try:
            slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_mat_cc_file(folder_name)
        except FileNotFoundError:
            continue
        nthebase4ever+=1
        slice_vol = find_slice(folder_name, '3_2_AxPasi7', slice_num,'thebase4ever')
        for mask in [mask_genu,mask_abody,mask_mbody,mask_pbody,mask_splenium]:
            parameter_val = calc_parameter_by_mask('median',slice_vol,mask)
            val_list.append(parameter_val)

    #protocol_list = ['HCP']*nhcp*5+['The Base']*nthebase*5+['The Base 4 Ever']*nthebase4ever*5
    protocol_list = ['\u0394 = 43.1[ms], \u03B4 = 10.6[ms], gmax = 10[G/cm]'] * nhcp * 5 + ['\u0394 = 60[ms], \u03B4 = 15.5[ms], gmax = 7.2[G/cm]'] * nthebase * 5 + ['\u0394 = 45[ms], \u03B4 = 15[ms], gmax = 7.9[G/cm]'] * nthebase4ever * 5
    parts_list = ['Genu','Anterior Body', 'Mid Body', 'Posterior Body', 'Splenium']*(nhcp+nthebase+nthebase4ever)
    #parts_list = [1,2,3,4,5] * (nhcp + nthebase + nthebase4ever)
    subji = [i for i in range(1,nhcp+nthebase+nthebase4ever+1)]*5
    subji.sort()
    d_vals = {'SubjNum':subji,'Protocol':protocol_list, 'CC_Part': parts_list,'ADD [\u03BCm]':val_list}
    cc_parts_table = pd.DataFrame(d_vals)
    cc_parts_table = detect_and_remove_outliers(cc_parts_table)[0]
    create_comperative_cc_vioplot(cc_parts_table, vio_type)
    #create_comperative_cc_boxplot(cc_parts_table)
    anova_for_different_protocols(cc_parts_table)


def detect_and_remove_outliers(table):
    from sklearn.neighbors import LocalOutlierFactor
    from statsmodels.robust.scale import mad

    if 'CC_Part' in table.columns:
        if 'Protocol' in table.columns:
            for protocol in set(table['Protocol']):
                #print(protocol)

                for part in set(table['CC_Part']):
                    #print(part)
                    vals = table['ADD [\u03BCm]'][table['CC_Part'] == part][table['Protocol'] == protocol].values
                    new_vals = vals.copy()
                    th = mad(vals)
                    diff = abs(vals - np.median(vals))
                    mask = diff / th > 2
                    #print(f'{vals} \n {mask}')
                    #new_vals[mask] = np.nan
                    new_vals[mask] = np.nanmean(new_vals[mask<1])
                    ii = table.loc[table['CC_Part'] == part][table['Protocol'] == protocol].index
                    table['ADD [\u03BCm]'][ii]=new_vals
                    print(sum(mask))

        else:
            for part in set(table['CC_Part']):
                vals = table['ADD [\u03BCm]'][table['CC_Part'] == part].values
                th = mad(vals)
                diff = abs(vals - np.median(vals))
                mask = diff / th > 2.5
                vals[mask] = np.nan
                table['ADD [\u03BCm]'][table['CC_Part'] == part] = vals
                #print(sum(mask))

        new_table = table
    else:
        lof = LocalOutlierFactor()
        numeric_table = table.select_dtypes(include='float64')
        detect_outlier = lof.fit_predict(numeric_table.values)
        mask = detect_outlier != -1
        new_table = table[mask]


    return new_table, sum(~mask)


def anova_for_cc_parts(table):
    from pingouin import rm_anova
    aov = rm_anova(data=table)
    print(f"F={aov['F'][0]},p={aov['p-unc'][0]}")
    print(aov)



def anova_for_different_protocols(table, posthoc=True):
    from scipy.stats import f_oneway
    from pingouin import rm_anova
    grouped_table = table.groupby('Protocol')
    for protocol in set(table['Protocol']):
        print(protocol)
        protable = grouped_table.get_group(protocol)
        #print(len(protable)/5)
        #f,p = f_oneway(protable['ADD [\u03BCm]'][protable['CC Part'] == 'Genu'].values,protable['ADD [\u03BCm]'][protable['CC Part'] == 'Anterior Body'].values,protable['ADD [\u03BCm]'][protable['CC Part'] == 'Mid Body'].values,protable['ADD [\u03BCm]'][protable['CC Part'] == 'Posterior Body'].values,protable['ADD [\u03BCm]'][protable['CC Part'] == 'Splenium'].values)
        #print(f'F={f}, p = {p}')
        #print(AnovaRM(data=protable, subject='SubjNum',depvar='ADD [\u03BCm]',within=['CC_Part']).fit())
        aov = rm_anova(data=protable, subject='SubjNum', dv='ADD [\u03BCm]', within=['CC_Part'])
        print(f"F={aov['F'][0]},p={aov['p-unc'][0]}")
        print(aov)
        if posthoc:
            from pingouin import pairwise_tests
            ptest = pairwise_tests(data=protable, subject='SubjNum', dv='ADD [\u03BCm]', within=['CC_Part'],padjust='bonf')
            print(ptest)

def compare_deltas_old_axcaliber(main_path,group,norm=False):
    subjD31, subjD45, subjD60 = choose_condition(main_path)

    deltas_list = ['D31 d18']*len(subjD31)*5+['D45 d13.2']*len(subjD45)*5+['D60 d11.3']*len(subjD60)*5
    parts_list = ['Genu','Anterior Body', 'Mid Body', 'Posterior Body', 'Splenium']*(len(subjD31)+len(subjD45)+len(subjD60))
    val_list = []
    for i,folder_name in enumerate(subjD31):
        slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_mat_cc_file(folder_name)
        slice_vol = find_slice(folder_name, '3_2_AxPasi7', slice_num,'thebase')
        for mask in [mask_genu,mask_abody,mask_mbody,mask_pbody,mask_splenium]:
            parameter_val = calc_parameter_by_mask('median',slice_vol,mask)
            val_list.append(parameter_val)

    for i,folder_name in enumerate(subjD45):
        slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_mat_cc_file(folder_name)
        slice_vol = find_slice(folder_name, '3_2_AxPasi7', slice_num,'thebase')
        for mask in [mask_genu,mask_abody,mask_mbody,mask_pbody,mask_splenium]:
            parameter_val = calc_parameter_by_mask('median',slice_vol,mask)
            val_list.append(parameter_val)

    for i,folder_name in enumerate(subjD60):
        slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_mat_cc_file(folder_name)
        slice_vol = find_slice(folder_name, '3_2_AxPasi7', slice_num,'thebase')
        for mask in [mask_genu,mask_abody,mask_mbody,mask_pbody,mask_splenium]:
            parameter_val = calc_parameter_by_mask('median',slice_vol,mask)
            val_list.append(parameter_val)
    if norm:
        for i,v in enumerate(val_list[0:len(val_list):5]):
            cc = val_list[i*5:i*5+5]
            cc = cc/cc[2]
            val_list[i*5:i*5 + 5] = cc
    d_vals = {'Protocol':deltas_list, 'CC part': parts_list,'ADD':val_list, 'Group':group}
    cc_parts_table = pd.DataFrame(d_vals)
    #cc_parts_table = detect_and_remove_outliers(cc_parts_table)[0]

    return cc_parts_table

def compare_group_study_OldAxCaliber(cc_parts_table, protocol):
    create_comperative_cc_vioplot(cc_parts_table[cc_parts_table['Protocol']==protocol],'sidebyside','Group')
    #create_comperative_cc_vioplot(cc_parts_table,'sidebyside')


if __name__ == '__main__':
    #show_vioplot_single_protocol('hcp')
    show_vioplot_compare_protocols(vio_type = 'sidebyside')
    #show_vioplot_compare_protocols(vio_type='split')

    # main_path = r'F:\Hila\Ax3D_Pack\V6\v7calibration\Old_AxCaliber\H'
    # cc_parts_table_H = compare_deltas_old_axcaliber(main_path, group='H')
    # main_path = r'F:\Hila\Ax3D_Pack\V6\v7calibration\Old_AxCaliber\MS'
    # cc_parts_table_MS = compare_deltas_old_axcaliber(main_path, group='MS')
    #
    # cc_parts_table = cc_parts_table_H.append(cc_parts_table_MS)
    # protocols = ['D31 d18', 'D45 d13.2', 'D60 d11.3']
    # compare_group_study_OldAxCaliber(cc_parts_table,protocols[0])



