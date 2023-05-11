import os, glob
import numpy as np
from network_analysis.global_network_properties import get_efficiency
from network_analysis.nodes_network_properties import get_local_efficiency
from network_analysis.norm_cm_by_atlas_areas import norm_mat
from scipy.stats import skew
import matplotlib.pyplot as plt

from network_analysis.norm_cm_by_random_mat import make_n_rand_mat, rand_mat

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
add_eff=[]
num_eff=[]
fa_eff=[]
add_eff_rand=[]
num_eff_rand=[]
fa_eff_rand=[]

add_loc_eff=[]
num_loc_eff=[]
fa_loc_eff=[]
add_loc_eff_rand=[]
num_loc_eff_rand=[]
fa_loc_eff_rand=[]

add_cost=[]
num_cost=[]
fa_cost=[]

th = 'HistMatch'
atlas = 'yeo7_200'
for sl in subj_list:

    add_cm = np.load(f'{sl}cm{os.sep}{atlas}_ADD_{th}_SC_cm_ord.npy')
    add_cm_rand = rand_mat(add_cm, 'links_shuffle')
    add_eff.append(get_efficiency(add_cm))
    add_eff_rand.append(get_efficiency(add_cm_rand))
    add_loc_eff.append(np.nanmean(get_local_efficiency(add_cm)))
    add_loc_eff_rand.append(np.nanmean(get_local_efficiency(add_cm_rand)))
    # np.fill_diagonal(add_cm, 0)
    # add_cm = add_cm/np.nanmax(add_cm)
    # add_cost.append(np.nansum(add_cm)/2)


    num_cm = np.load(f'{sl}cm{os.sep}{atlas}_Num_{th}_SC_cm_ord.npy')
    num_eff.append(get_efficiency(num_cm))
    num_eff_rand.append(get_efficiency(add_cm_rand))
    num_loc_eff.append(np.nanmean(get_local_efficiency(num_cm)))
    num_loc_eff_rand.append(np.nanmean(get_local_efficiency(add_cm_rand)))
    # np.fill_diagonal(num_cm, 0)
    # num_cm = num_cm/np.nanmax(num_cm)
    # num_cost.append(np.nansum(num_cm)/2)

    fa_cm = np.load(f'{sl}cm{os.sep}{atlas}_FA_{th}_SC_cm_ord.npy')
    fa_eff.append(get_efficiency(fa_cm))
    fa_eff_rand.append(get_efficiency(add_cm_rand))
    fa_loc_eff.append(np.nanmean(get_local_efficiency(fa_cm)))
    fa_loc_eff_rand.append(np.nanmean(get_local_efficiency(add_cm_rand)))
    # np.fill_diagonal(fa_cm, 0)
    # fa_cm = fa_cm/np.nanmax(fa_cm)
    # fa_cost.append(np.nansum(fa_cm)/2)

# plt.scatter(num_cost, num_eff, color=[0.2, 0.7, 0.6], s=10)
# plt.scatter(add_cost, add_eff, color=[0.8, 0.5, 0.3], s=10)
# plt.scatter(fa_cost, fa_eff, color=[0.3, 0.3, 0.5], s=10)
#
# plt.legend(['Num', 'ADD', 'FA'])
# plt.xlabel('Cost')
# plt.ylabel('Global Efficiency')
# plt.show()
#
# plt.scatter(num_cost, num_loc_eff, color=[0.2, 0.7, 0.6], s=10)
# plt.scatter(add_cost, add_loc_eff, color=[0.8, 0.5, 0.3], s=10)
# plt.scatter(fa_cost, fa_loc_eff, color=[0.3, 0.3, 0.5], s=10)
#
# plt.legend(['Num', 'ADD', 'FA'])
# plt.xlabel('Cost')
# plt.ylabel('Local Efficiency')
# plt.show()

plt.scatter(num_loc_eff,num_eff, color=[0.2, 0.7, 0.6], s=10)
plt.scatter(add_loc_eff,add_eff, color=[0.8, 0.5, 0.3], s=10)
plt.scatter(fa_loc_eff, fa_eff, color=[0.3, 0.3, 0.5], s=10)

plt.legend(['Num', 'ADD', 'FA'])
plt.ylabel('Global Efficiency')
plt.xlabel('Local Efficiency')
plt.show()


plt.scatter(num_loc_eff/num_loc_eff_rand,num_eff/num_eff_rand, color=[0.2, 0.7, 0.6], s=10)
plt.scatter(add_loc_eff/add_loc_eff_rand,add_eff/add_eff_rand, color=[0.8, 0.5, 0.3], s=10)
plt.scatter(fa_loc_eff/fa_loc_eff_rand, fa_eff/fa_eff_rand, color=[0.3, 0.3, 0.5], s=10)

plt.legend(['norm Num', 'norm ADD', 'norm FA'])
plt.ylabel('Global Efficiency')
plt.xlabel('Local Efficiency')
plt.show()

