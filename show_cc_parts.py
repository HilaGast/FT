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


def show_cc_parts(folder_name,weight_by,n,nii_file,hue,saturation,scale):

    tract_path = folder_name + r'\streamlines' + n + '_CCMid_mct001rt20_msmt_5d.trk'
    ccmid = load_ft(tract_path,nii_file)

    tract_path = folder_name + r'\streamlines' + n + '_CC_ForcepsMajor_mct001rt20_msmt_5d.trk'
    ccfmaj = load_ft(tract_path,nii_file)

    tract_path = folder_name + r'\streamlines' + n + '_CC_ForcepsMinor_mct001rt20_msmt_5d.trk'
    ccfmin = load_ft(tract_path,nii_file)


    weight_by_data, affine = load_weight_by_img(folder_name, weight_by)

    mean_ccmid = mean_vol_bundle(weight_by_data,ccmid,affine)
    print(f'mean CCMid: {mean_ccmid[0]}')
    mean_ccfmaj = mean_vol_bundle(weight_by_data,ccfmaj,affine)
    print(f'mean CC_ForcepsMajor: {mean_ccfmaj[0]}')
    mean_ccfmin = mean_vol_bundle(weight_by_data,ccfmin,affine)
    print(f'mean CC_ForcepsMinor: {mean_ccfmin[0]}')


    cc = ccmid
    cc.extend(ccfmaj)
    cc.extend(ccfmin)

    cc_vols = mean_ccmid
    cc_vols.extend(mean_ccfmaj)
    cc_vols.extend(mean_ccfmin)

    cc = transform_streamlines(cc, np.linalg.inv(affine))
    show_tracts(hue ,saturation ,scale ,cc ,cc_vols ,folder_name ,'_cc_'+weight_by+'_mean_parts')

if __name__ == '__main__':
    weight_by = '1.5_2.5_AxPasi7'
    hue=[0.25, -0.05] #hot
    saturation = [0.0, 1.0]
    scale = [5,10]
    subj = all_subj_folders
    names = all_subj_names


    for s, n in zip(subj[8:9], names[8:9]):
        folder_name = subj_folder + s
        dir_name = folder_name + '\streamlines'
        gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
    bval_file = bvec_file[:-4:] + 'bval'

    show_cc_parts(folder_name,weight_by,n,nii_file,hue,saturation,scale)