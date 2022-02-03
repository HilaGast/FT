import os, glob
import numpy as np
from network_analysis.global_network_properties import get_rich_club_curve
from network_analysis.norm_cm_by_atlas_areas import norm_mat
#shortlist = ['H:\\HCP\\123925\\', 'H:\\HCP\\193441\\', 'H:\\HCP\\304727\\', 'H:\\HCP\\204016\\', 'H:\\HCP\\297655\\', 'H:\\HCP\\186848\\', 'H:\\HCP\\180432\\', 'H:\\HCP\\555348\\', 'H:\\HCP\\283543\\', 'H:\\HCP\\181232\\', 'H:\\HCP\\387959\\', 'H:\\HCP\\613538\\', 'H:\\HCP\\153227\\', 'H:\\HCP\\414229\\', 'H:\\HCP\\753251\\', 'H:\\HCP\\115017\\', 'H:\\HCP\\248339\\', 'H:\\HCP\\168947\\', 'H:\\HCP\\349244\\', 'H:\\HCP\\547046\\', 'H:\\HCP\\199453\\', 'H:\\HCP\\110411\\', 'H:\\HCP\\886674\\', 'H:\\HCP\\113922\\', 'H:\\HCP\\151223\\', 'H:\\HCP\\130518\\', 'H:\\HCP\\380036\\', 'H:\\HCP\\671855\\', 'H:\\HCP\\286650\\', 'H:\\HCP\\158136\\', 'H:\\HCP\\194443\\', 'H:\\HCP\\154734\\', 'H:\\HCP\\106824\\', 'H:\\HCP\\880157\\', 'H:\\HCP\\111009\\', 'H:\\HCP\\299154\\', 'H:\\HCP\\599469\\', 'H:\\HCP\\530635\\', 'H:\\HCP\\609143\\', 'H:\\HCP\\130922\\', 'H:\\HCP\\245333\\', 'H:\\HCP\\513130\\', 'H:\\HCP\\154229\\', 'H:\\HCP\\178647\\', 'H:\\HCP\\192439\\', 'H:\\HCP\\395251\\', 'H:\\HCP\\112314\\', 'H:\\HCP\\168240\\', 'H:\\HCP\\150928\\', 'H:\\HCP\\884064\\', 'H:\\HCP\\180129\\', 'H:\\HCP\\792867\\', 'H:\\HCP\\517239\\', 'H:\\HCP\\106319\\', 'H:\\HCP\\481951\\', 'H:\\HCP\\628248\\', 'H:\\HCP\\522434\\', 'H:\\HCP\\123420\\', 'H:\\HCP\\129331\\', 'H:\\HCP\\397760\\', 'H:\\HCP\\951457\\', 'H:\\HCP\\157336\\', 'H:\\HCP\\104820\\', 'H:\\HCP\\118730\\', 'H:\\HCP\\701535\\', 'H:\\HCP\\550439\\', 'H:\\HCP\\176744\\', 'H:\\HCP\\871762\\', 'H:\\HCP\\715647\\', 'H:\\HCP\\151627\\', 'H:\\HCP\\707749\\', 'H:\\HCP\\195849\\', 'H:\\HCP\\206727\\', 'H:\\HCP\\336841\\', 'H:\\HCP\\197348\\', 'H:\\HCP\\188145\\', 'H:\\HCP\\192237\\', 'H:\\HCP\\191437\\', 'H:\\HCP\\139435\\', 'H:\\HCP\\663755\\', 'H:\\HCP\\158843\\', 'H:\\HCP\\118124\\', 'H:\\HCP\\469961\\', 'H:\\HCP\\894673\\', 'H:\\HCP\\872158\\', 'H:\\HCP\\765864\\', 'H:\\HCP\\203923\\', 'H:\\HCP\\134829\\', 'H:\\HCP\\111716\\', 'H:\\HCP\\742549\\', 'H:\\HCP\\144933\\', 'H:\\HCP\\286347\\', 'H:\\HCP\\130417\\', 'H:\\HCP\\911849\\', 'H:\\HCP\\263436\\', 'H:\\HCP\\102614\\', 'H:\\HCP\\101107\\', 'H:\\HCP\\120414\\', 'H:\\HCP\\210415\\', 'H:\\HCP\\214019\\', 'H:\\HCP\\180937\\', 'H:\\HCP\\113215\\', 'H:\\HCP\\121719\\', 'H:\\HCP\\248238\\', 'H:\\HCP\\727654\\', 'H:\\HCP\\352132\\', 'H:\\HCP\\627549\\', 'H:\\HCP\\158540\\', 'H:\\HCP\\212217\\', 'H:\\HCP\\307127\\', 'H:\\HCP\\378756\\', 'H:\\HCP\\138837\\', 'H:\\HCP\\381038\\', 'H:\\HCP\\117021\\', 'H:\\HCP\\877168\\', 'H:\\HCP\\134021\\', 'H:\\HCP\\159138\\']

shortlist = glob.glob(f'F:\data\V7\HCP\*{os.sep}')
max_k= 50
add_rc=[]
num_rc=[]
for sl in shortlist:

    add_cm = np.load(f'{sl}cm{os.sep}add_bna_cm_ord.npy')
    add_rc.append(get_rich_club_curve(add_cm, max_k))

    num_cm = np.load(f'{sl}cm{os.sep}num_bna_cm_ord_corrected.npy')
    #num_cm = norm_mat(sl,num_cm,'bna')
    num_rc.append(get_rich_club_curve(num_cm, max_k))


num_rc = np.asarray(num_rc)
num_rc = num_rc[:,1:]
add_rc = np.asarray(add_rc)
add_rc = add_rc[:,1:]

num_rc_mean = np.nanmean(num_rc, axis=0)
add_rc_mean = np.nanmean(add_rc, axis=0)
k = list(range(1,max_k))

import matplotlib.pyplot as plt

plt.plot(k,num_rc_mean,color=[0.2, 0.7, 0.6])
plt.plot(k, add_rc_mean,color=[0.2, 0.5, 0.8])
plt.show()
