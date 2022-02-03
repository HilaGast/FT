import os, glob
import numpy as np
from network_analysis.norm_cm_by_random_mat import make_n_rand_mat
from network_analysis.global_network_properties import get_rich_club_curve

shortlist = glob.glob(f'F:\data\V7\HCP\*{os.sep}')

all_add_mat = []
all_num_mat = []
cerbellum_i = [i for i in range(123,151)]
for sl in shortlist:

    add_cm = np.load(f'{sl}cm{os.sep}add_bna_cm_ord.npy')
    add_cm = np.delete(add_cm,cerbellum_i, axis=0)
    add_cm = np.delete(add_cm,cerbellum_i, axis=1)
    all_add_mat.append(add_cm)

    num_cm = np.load(f'{sl}cm{os.sep}num_bna_cm_ord_corrected.npy')
    num_cm = np.delete(num_cm,cerbellum_i, axis=0)
    num_cm = np.delete(num_cm,cerbellum_i, axis=1)
    all_num_mat.append(num_cm)

all_add_mat = np.asarray(all_add_mat)
all_num_mat = np.asarray(all_num_mat)

mask_mat = all_num_mat > 0
mask_mat = np.sum(mask_mat,axis=0)/mask_mat.shape[0]
mask_mat = mask_mat>0.75 #choose only edges that appeared in >75% of subjects

add = all_add_mat.sum(0)/(all_add_mat != 0).sum(0) #mean of nonzeros
num = all_num_mat.sum(0)/(all_num_mat != 0).sum(0) #mean of nonzeros

masked_add = add * mask_mat
masked_num = num * mask_mat

masked_add[np.isnan(masked_add)] = 0
masked_num[np.isnan(masked_num)] = 0

#norm_add = masked_add/masked_add.max() #normalize between 0 and 1
#norm_num = masked_num/masked_num.max() #normalize between 0 and 1

n=1000
rand_add = make_n_rand_mat(masked_add,n,'links shuffle')
rand_num = make_n_rand_mat(masked_num,n,'links shuffle')

max_k = 20
rand_num_rc = []
rand_add_rc = []

for i in range(0,n):

    rand_num_rc.append(get_rich_club_curve(rand_num[:,:,i],max_k))
    rand_add_rc.append(get_rich_club_curve(rand_add[:,:,i],max_k))


mean_rand_num_rc = np.asarray(rand_num_rc)
mean_rand_num_rc = mean_rand_num_rc[:,1:]
mean_rand_add_rc = np.asarray(rand_add_rc)
mean_rand_add_rc  = mean_rand_add_rc [:,1:]

mean_rand_num_rc = np.nanmean(mean_rand_num_rc, axis=0)
mean_rand_add_rc  = np.nanmean(mean_rand_add_rc, axis=0)
k = list(range(1,max_k))

# Option 1 = using average mat:
num_rc = get_rich_club_curve(masked_num,max_k)
add_rc = get_rich_club_curve(masked_add,max_k)

num_rc = num_rc[1:]
add_rc = add_rc[1:]


# Option 2 = using subjects mat:
#num_rc = []
#add_rc = []
#for i in range(0,all_num_mat.shape[0]):
#    num_rc.append(get_rich_club_curve(all_num_mat[i,:,:],max_k))
#    add_rc.append(get_rich_club_curve(all_add_mat[i,:,:],max_k))

#num_rc = np.asarray(num_rc)
#add_rc = np.asarray(add_rc)
#num_rc = num_rc[:,1:]
#add_rc = add_rc[:,1:]

#num_rc = np.nanmean(num_rc,axis=0)
#add_rc = np.nanmean(add_rc,axis=0)






import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.set_xlabel('K')
ax1.set_ylabel('\u03A6 (RC coefficient)')
ax1.plot(k,num_rc,color=[0.2, 0.7, 0.6])
ax1.plot(k, mean_rand_num_rc,color=[0.2, 0.5, 0.8])

ax2 = ax1.twinx()
ax2.set_ylabel('\u03A6 Norm')
ax2.plot(k,num_rc/mean_rand_num_rc, color = [0.5, 0.3, 0.8])
fig.legend(['RC','RC rand', 'RC norm'],loc=4)
plt.title('Num')
fig.tight_layout()
plt.show()


fig, ax1 = plt.subplots()
ax1.set_xlabel('K')
ax1.set_ylabel('\u03A6 (RC coefficient)')
ax1.plot(k,add_rc,color=[0.2, 0.7, 0.6])
ax1.plot(k, mean_rand_add_rc,color=[0.2, 0.5, 0.8])

ax2 = ax1.twinx()
ax2.set_ylabel('\u03A6 Norm')
ax2.plot(k,add_rc/mean_rand_add_rc, color = [0.5, 0.3, 0.8])
fig.legend(['RC','RC rand', 'RC norm'],loc=4)
plt.title('ADD')
fig.tight_layout()
plt.show()
