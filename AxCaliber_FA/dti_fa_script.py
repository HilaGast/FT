
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy
from folder_organization.organizeTheBase import *
from dipy.core.gradients import gradient_table
import nibabel as nib
all_subj_folders = glob(r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep\post_covid19\post\*')
for s in all_subj_folders:
    if not os.listdir(f'{s}\dti'):
        continue
    for file in os.listdir(os.path.join(s,'dti')):
        if '.nii' in file:
            dti_file = os.path.join(s,'dti',file)
            bvec_file = os.path.join(s,'dti',file[:-3])+'bvec'
            bval_file = os.path.join(s,'dti',file[:-3])+'bval'
            continue
    gtab = gradient_table(bval_file, bvec_file, small_delta=15.5)
    dtid = nib.load(dti_file)
    data = dtid.get_fdata()

    tensor_model = dti.TensorModel(gtab)
    tenfit = tensor_model.fit(data)
    fa = fractional_anisotropy(tenfit.evals)
    fa_file = nib.Nifti1Image(fa,dtid.affine,dtid.header)
    fa_name = os.path.join(s,'dti','dti_fa.nii')
    nib.save(fa_file,fa_name)