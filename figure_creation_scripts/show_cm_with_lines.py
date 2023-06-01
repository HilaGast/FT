import numpy as np
import matplotlib.pyplot as plt

def creating_triangles(ax,limits, color, linewidth):
    add = linewidth/5
    ax.plot([limits[0]-add,limits[1]+add],[limits[1]+add,limits[1]+add], color=color, linewidth=linewidth)
    ax.plot([limits[1]+add,limits[1]+add],[limits[0]-add,limits[1]+add], color=color, linewidth=linewidth)
    ax.plot([limits[0]-add,limits[1]+add],[limits[0]-add,limits[0]-add], color=color, linewidth=linewidth)
    ax.plot([limits[0]-add,limits[0]-add],[limits[0]-add,limits[1]+add], color=color, linewidth=linewidth)

if "__main__" == __name__:
    mat_type = 'time_th3'
    # Load
    tdi_LH = np.load(fr'G:\data\V7\HCP\cm\median_yeo7_100_{mat_type}_Org_SC_LH.npy')
    tdi_RH = np.load(fr'G:\data\V7\HCP\cm\median_yeo7_100_{mat_type}_Org_SC_RH.npy')

    # revert tdi
    tdi_LH = np.nanmax(tdi_LH) - tdi_LH
    tdi_RH = np.nanmax(tdi_RH) - tdi_RH

    # Draw matrix:
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(tdi_LH)
    ax[1].imshow(tdi_RH)
    fig.colorbar(ax[0].imshow(tdi_LH), ax=ax[0])
    ax[0].set_title('TDI LH')
    ax[1].set_title('TDI RH')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    # Draw lines:
    # Vis:
    # LH
    creating_triangles(ax[0],[0,8], 'red', 2)
    # RH
    creating_triangles(ax[1],[0,7], 'red', 2)
    # SomMot:
    # LH
    creating_triangles(ax[0],[9,14], 'red', 2)
    # RH
    creating_triangles(ax[1],[8,15], 'red', 2)
    # DorsAttn:
    # LH
    creating_triangles(ax[0],[15,22], 'red', 2)
    # RH
    creating_triangles(ax[1],[16,22], 'red', 2)
    # SalVentAttn:
    # LH
    creating_triangles(ax[0],[23,29], 'red', 2)
    # RH
    creating_triangles(ax[1],[23,27], 'red', 2)
    # Limbic:
    # LH
    creating_triangles(ax[0],[30,32], 'red', 2)
    # RH
    creating_triangles(ax[1],[28,29], 'red', 2)
    # Cont:
    # LH
    creating_triangles(ax[0],[33,36], 'red', 2)
    # RH
    creating_triangles(ax[1],[30,38], 'red', 2)
    # Default:
    # LH
    creating_triangles(ax[0],[37,49], 'red', 2)#
    # RH
    creating_triangles(ax[1],[39,49], 'red', 2)
    plt.show()
