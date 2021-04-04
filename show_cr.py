from weighted_tracts import *
from dipy.tracking.streamline import values_from_volume
from dipy.tracking.streamline import transform_streamlines


def mean_vol_bundle(weight_by_data,bundle,affine):
    vol_per_tract = values_from_volume(weight_by_data, bundle, affine=affine)
    vs=[]
    for st in vol_per_tract:
        vs.append(np.nanmedian(st))
    mean_bundle = [np.nanmean(vs)]
    mean_bundle = mean_bundle*len(vol_per_tract)


    return mean_bundle


def show_cr_parts(folder_name,side,weight_by,n,nii_file,hue,saturation,scale):

    tract_path = folder_name + r'\streamlines' + n + '_OPT_'+side+'_mct001rt20_msmt_5d.trk'
    opt = load_ft(tract_path,nii_file)

    tract_path = folder_name + r'\streamlines' + n + '_FPT_'+side+'_mct001rt20_msmt_5d.trk'
    fpt = load_ft(tract_path,nii_file)

    tract_path = folder_name + r'\streamlines' + n + '_PPT_'+side+'_mct001rt20_msmt_5d.trk'
    ppt = load_ft(tract_path,nii_file)


    tract_path = folder_name + r'\streamlines' + n + '_CST_'+side+'_mct001rt20_msmt_5d.trk'
    cst = load_ft(tract_path,nii_file)

    weight_by_data, affine = load_weight_by_img(folder_name, weight_by)

    mean_opt = mean_vol_bundle(weight_by_data,opt,affine)
    print(f'mean opt: {mean_opt[0]}')
    mean_fpt = mean_vol_bundle(weight_by_data,fpt,affine)
    print(f'mean fpt: {mean_fpt[0]}')
    mean_ppt = mean_vol_bundle(weight_by_data,ppt,affine)
    print(f'mean ppt: {mean_ppt[0]}')
    mean_cst = mean_vol_bundle(weight_by_data,cst,affine)
    print(f'mean cst: {mean_cst[0]}')

    cr = opt
    cr.extend(fpt)
    cr.extend(ppt)
    cr.extend(cst)

    cr_vols = mean_opt
    cr_vols.extend(mean_fpt)
    cr_vols.extend(mean_ppt)
    cr_vols.extend(mean_cst)

    cr = transform_streamlines(cr, np.linalg.inv(affine))
    show_tracts(hue ,saturation ,scale ,cr ,cr_vols ,folder_name ,'_cr_'+side+weight_by+'_mean_parts')

if __name__ == '__main__':
    side = 'L'
    weight_by = '1_3_AxPasi7'
    hue=[0.25, -0.05] #hot
    saturation = [0.0, 1.0]
    scale = [6,8.5]
    subj = all_subj_folders
    names = all_subj_names


    for s, n in zip(subj[9:10], names[9:10]):
        folder_name = subj_folder + s
        dir_name = folder_name + '\streamlines'
        gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
    bval_file = bvec_file[:-4:] + 'bval'

    show_cr_parts(folder_name,side,weight_by,n,nii_file,hue,saturation,scale)