import numpy as np
import nibabel as nib


def bval_vec(bval_file):
    from dipy.io import read_bvals_bvecs

    bvals = read_bvals_bvecs(bval_file,None)[0]
    bvals = np.around(bvals, decimals=-1)
    b0i = np.where(bvals==0)[0]

    return bvals, b0i

def qvec_calc(bvals,bmax,gmax,delta):
    q = 4257*delta*gmax*bvals/bmax

    return q

def load_diffusion_file(nii_file_name):
    hardi_img = nib.load(nii_file_name)
    data = hardi_img.get_fdata()
    data[data<0]=0
    data = data[:,:,5,:]

    return data


def load_cc_mask(cc_file):
    import scipy.io as sio
    cc_mask_mat = sio.loadmat(cc_file)['mask_cc']

    return cc_mask_mat


def b0_mean_value(data, b0i, cc_mask):
    b0_vals = []
    for s in b0i:
        mat = data[:,:,s]
        mat = mat[cc_mask]
        b0_vals.append(np.mean(mat[mat>0]))

    b0_decay = np.mean(b0_vals)

    return b0_decay


def decay_vec(data, cc_mask, b0_decay):
    b_vals = []
    for s in range(data.shape[2]):
        mat = data[:, :, s]
        mat = mat[cc_mask]
        b_vals.append(np.mean(mat[mat>0]))

    bvals_decay = b_vals/b0_decay
    bvals_decay = bvals_decay/np.max(bvals_decay)

    return bvals_decay


def draw_heatmap_H11(bvals, D75, D65, D55, D45, D35):
    import matplotlib.pyplot as plt
    D = [r'$\Delta75$',r'$\Delta65$',r'$\Delta55$',r'$\Delta45$',r'$\Delta35$']
    bvals_names = [str(int(b)) for b in bvals]
    bvals_names[0]=''
    bvals_names[1] = ''
    bvals_names[3] = ''
    bvals_names[4] = ''

    fig, ax = plt.subplots()
    im = ax.imshow([D75,D65,D55,D45,D35],cmap='hot')
    ax.set_xticks(np.arange(len(bvals_names)))
    ax.set_yticks(np.arange(len(D)))
    ax.set_xticklabels(bvals_names)
    ax.set_yticklabels(D)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    plt.colorbar(im, shrink=0.2)
    plt.show()

def compare_Deltas_H11():
    import os

    main_folder = r'C:\Users\Admin\Desktop\v7_calibration\Delta effect\YA_lab_Shani_001278_20161207_1122\H011'

    cc_file = os.path.join(main_folder,'mask_CC_slice6.mat')
    cc_mask = load_cc_mask(cc_file)

    Delta_file_name = 'H011_26_4000_75_18.3_124'
    bval_file = os.path.join(main_folder, Delta_file_name+'.bval')
    nii_file_name = os.path.join(main_folder,Delta_file_name+'.nii')
    bvals, b0i = bval_vec(bval_file)
    data = load_diffusion_file(nii_file_name)
    b0_D75 = b0_mean_value(data, b0i, cc_mask)
    D75 = decay_vec(data, cc_mask, b0_D75)

    Delta_file_name = 'H011_25_4000_65_18.3_119'
    nii_file_name = os.path.join(main_folder,Delta_file_name+'.nii')
    data = load_diffusion_file(nii_file_name)
    b0_D65 = b0_mean_value(data, b0i, cc_mask)
    D65 = decay_vec(data, cc_mask, b0_D65)

    Delta_file_name = 'H011_24_4000_55_18.3_113'
    nii_file_name = os.path.join(main_folder,Delta_file_name+'.nii')
    data = load_diffusion_file(nii_file_name)
    b0_D55 = b0_mean_value(data, b0i, cc_mask)
    D55 = decay_vec(data, cc_mask, b0_D55)

    Delta_file_name = 'H011_23_4000_45_18.3_110'
    nii_file_name = os.path.join(main_folder,Delta_file_name+'.nii')
    data = load_diffusion_file(nii_file_name)
    b0_D45 = b0_mean_value(data, b0i, cc_mask)
    D45 = decay_vec(data, cc_mask, b0_D45)

    Delta_file_name = 'H011_22_4000_35_18.3_107'
    nii_file_name = os.path.join(main_folder,Delta_file_name+'.nii')
    data = load_diffusion_file(nii_file_name)
    b0_D35 = b0_mean_value(data, b0i, cc_mask)
    D35 = decay_vec(data, cc_mask, b0_D35)

    #draw_heatmap_H11(bvals, D75, D65, D55, D45, D35)
    return bvals, D75, D65, D55, D45, D35


def compare_Deltas_qbased():
    import os

    main_folder = r'C:\Users\Admin\Desktop\v7_calibration\Delta effect\YA_lab_Assi_001270_20160922_1455\H009'
    cc_file = os.path.join(main_folder,'mask_CC_slice6.mat')
    cc_mask = load_cc_mask(cc_file)

    Delta_file_name = 'H009_7_9020_101_15.5_138'
    bval_file = os.path.join(main_folder, Delta_file_name+'.bval')
    nii_file_name = os.path.join(main_folder,Delta_file_name+'.nii')
    bvals, b0i = bval_vec(bval_file)
    q101 = qvec_calc(bvals,bvals[-1],7.4,15.5)
    data = load_diffusion_file(nii_file_name)
    b0_D101 = b0_mean_value(data, b0i, cc_mask)
    D101 = decay_vec(data, cc_mask, b0_D101)

    Delta_file_name = 'H009_6_6520_75_15.5_111'
    bval_file = os.path.join(main_folder, Delta_file_name+'.bval')
    nii_file_name = os.path.join(main_folder,Delta_file_name+'.nii')
    bvals, b0i = bval_vec(bval_file)
    q75 = qvec_calc(bvals,bvals[-1],7.37,15.5)
    data = load_diffusion_file(nii_file_name)
    b0_D75 = b0_mean_value(data, b0i, cc_mask)
    D75 = decay_vec(data, cc_mask, b0_D75)

    Delta_file_name = 'H009_5_4170_50_15.5_86'
    bval_file = os.path.join(main_folder, Delta_file_name+'.bval')
    nii_file_name = os.path.join(main_folder,Delta_file_name+'.nii')
    bvals, b0i = bval_vec(bval_file)
    q50 = qvec_calc(bvals,bvals[-1],7.35,15.5)
    data = load_diffusion_file(nii_file_name)
    b0_D50 = b0_mean_value(data, b0i, cc_mask)
    D50 = decay_vec(data, cc_mask, b0_D50)

    Delta_file_name = 'H009_4_1770_25_15.5_63'
    bval_file = os.path.join(main_folder, Delta_file_name+'.bval')
    nii_file_name = os.path.join(main_folder,Delta_file_name+'.nii')
    bvals, b0i = bval_vec(bval_file)
    q25 = qvec_calc(bvals,bvals[-1],7.2,15.5)
    data = load_diffusion_file(nii_file_name)
    b0_D25 = b0_mean_value(data, b0i, cc_mask)
    D25 = decay_vec(data, cc_mask, b0_D25)


def bi_exp_func(x,params):
    y = params[0]*np.exp(-x*params[2])+params[1]*np.exp(-x*params[3])

    return y

def const(params):
    pen = 1-params[0]-params[1]

    return pen

def func_2_min(params, x, y):
    y_pred = bi_exp_func(x,params)
    new_y = np.sum((y_pred-y)**2)

    return new_y

def find_parameters_vals(bvals,bvals_decay):
    #from scipy.optimize import curve_fit
    #params = curve_fit(bi_exp_func,bvals,bvals_decay,p0=[0.3,0.7,0.001,0.0001])

    from scipy.optimize import minimize
    cons = {'type':'eq', 'fun': const}
    params = minimize(func_2_min,x0=[0.2,0.8,0.01,0.001], args=(bvals,bvals_decay), constraints = cons).x

    return params

def plot_bi_exp_params(bvals, D75, D65, D55, D45, D35):
    import matplotlib.pyplot as plt

    params75 = find_parameters_vals(bvals, D75)
    params65 = find_parameters_vals(bvals, D65)
    params55 = find_parameters_vals(bvals, D55)
    params45 = find_parameters_vals(bvals, D45)
    params35 = find_parameters_vals(bvals, D35)

    x = np.sqrt(2*np.array([75, 65, 55, 45, 35]))

    y1 = x*np.sqrt([min(params75[2],params75[3]), min(params65[2],params65[3]), min(params55[2],params55[3]), min(params45[2],params45[3]), min(params35[2],params35[3])])
    y2 = x*np.sqrt([max(params75[2],params75[3]), max(params65[2],params65[3]), max(params55[2],params55[3]), max(params45[2],params45[3]), max(params35[2],params35[3])])

    plt.plot(x, y1, '*r')

    plt.plot(x, y2, '*b')

    plt.show()

    y35 = bi_exp_func(bvals, params35)
    y45 = bi_exp_func(bvals, params45)
    y55 = bi_exp_func(bvals, params55)
    y65 = bi_exp_func(bvals, params65)
    y75 = bi_exp_func(bvals, params75)

    plt.plot(bvals, y75, 'b')
    plt.plot(bvals, D75, '*b')
    plt.plot(bvals, y65, 'r')
    plt.plot(bvals, D65, '*r')
    plt.plot(bvals, y55, 'g')
    plt.plot(bvals, D55, '*g')
    plt.plot(bvals, y45, 'm')
    plt.plot(bvals, D45, '*m')
    plt.plot(bvals, y35, 'c')
    plt.plot(bvals, D35, '*c')
    plt.legend(['fitted $\Delta75$', '$\Delta75$', 'fitted $\Delta65$', '$\Delta65$', 'fitted $\Delta55$', '$\Delta55$',
                'fitted $\Delta45$', '$\Delta45$', 'fitted $\Delta35$', '$\Delta35$'])
    plt.show()

