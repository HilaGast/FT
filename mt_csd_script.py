from dmipy.dmipy.optimizers_fod.csd_cvxpy import CsdCvxpyOptimizer

from dipy.core.gradients import gradient_table
from dmipy.dmipy.core.acquisition_scheme import gtab_dipy2dmipy
#gtab_dipy = gradient_table(bvalues, gradient_directions, big_delta=Delta, small_delta=delta)
#acq_scheme_mipy = gtab_dipy2dmipy(gtab_dipy)
#acq_scheme_mipy.print_acquisition_info

# load the necessary modules
from dmipy.dmipy.core import modeling_framework
from dmipy.dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from os.path import join
import numpy as np

'''Setting up a Dmipy acquisition scheme by loading acquisition parameters:
To set up an acquisition scheme directly from b-values one can load them directly from text files.
The b-values and gradient directions, along with the used pulse duration time
delta and pulse separation time Delta of the acquisition must be known. 
The dmipy toolbox uses SI units, so be careful, as bvalues are typically saved in s/mm^2, 
but as a rule we need them in s/m^2.

As an example we load the acquisition parameters of the WU-MINN Human Connectome Project:
'''
import nibabel as nib
import os
import numpy as np
subj_name = "C150"
path_name=r'C:\Users\Admin\my_scripts\C1156_50\for_FT\C133'

file_name=r'20190411_182642ep2dd155D60MB3APs005a001.nii'
charmed_file_name = os.path.join(path_name,file_name)
hardi_img = nib.load(charmed_file_name)
data = hardi_img.get_data()
bval_name = r'20190411_182642ep2dd155D60MB3APs005a001.bval'
bvec_name = r'20190411_182642ep2dd155D60MB3APs005a001.bvec'
bval_file = os.path.join(path_name,bval_name)
bvec_file = os.path.join(path_name,bvec_name)
# we can then load the parameters themselves and convert them to SI units:
bvalues = np.loadtxt(bval_file)  # given in s/mm^2. An ndarray of (n,1) length.
bvalues_SI = bvalues * 1e6  # now given in SI units as s/m^2
gradient_directions = np.loadtxt(bvec_file)  # on the unit sphere. An ndarray of (n,3) shape.
gradient_directions = gradient_directions.transpose()

# The delta and Delta times we know from the HCP documentation in seconds
delta = 0.0155
Delta = 0.0592

'''Creating Acquisition Scheme:'''
# The acquisition scheme we use in the toolbox is then created as follows:
acq_scheme = acquisition_scheme_from_bvalues(bvalues_SI, gradient_directions, delta, Delta)

acq_scheme.print_acquisition_info
from dmipy.dmipy.tissue_response.three_tissue_response import three_tissue_response_dhollander16

wm, gm, csf, all = three_tissue_response_dhollander16(acq_scheme, data, wm_algorithm='tournier13',
        wm_N_candidate_voxels=300, gm_perc=0.02, csf_perc=0.1)

tissue_responses = [wm, gm, csf]

''' Creating the MT-CSD model:'''
from dmipy.dmipy.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel
mt_csd_mod = MultiCompartmentSphericalHarmonicsModel(tissue_responses)

fit_args = {'acquisition_scheme': acq_scheme, 'data': data, 'mask': data[..., 0]>0}

mt_csd_fits = []
for fit_S0_response in [True, False]:
    mt_csd_fits.append(mt_csd_mod.fit(fit_S0_response=fit_S0_response, **fit_args))