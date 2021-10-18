import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 


def plot_result(source_map, predicted_target_img, validation_map, fig_size=(16, 4)):
    # cmap = "Spectral"


    vmin = np.min([0.0, np.min(source_map), np.min(predicted_target_img), np.min(validation_map)])
    allmaxs = [np.max(source_map), np.max(predicted_target_img), np.max(validation_map)]
    maxindex = np.argmax(allmaxs)
    vmax = allmaxs[maxindex]

    eps = 0.05

    f, axarr = plt.subplots(1, 3, figsize=fig_size, num=102) 

    a =[]
    a.append( axarr[0].matshow(source_map+eps, vmin=eps, vmax=vmax, norm=LogNorm(vmin=eps, vmax=vmax)) )

    a.append( axarr[1].matshow(predicted_target_img+eps, vmin=eps, vmax=vmax, norm=LogNorm(vmin=eps, vmax=vmax)) )

    a.append( axarr[2].matshow(validation_map+eps, vmin=eps, vmax=vmax, norm=LogNorm(vmin=eps, vmax=vmax)) )

    titles = ['Source', 'Predicted Target', 'Target']

    for i, ax in enumerate(axarr):
        ax.set_axis_off()
        ax.set_title(titles[i])

    f.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.4, hspace=0.1)
    cb_ax = f.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = f.colorbar(a[maxindex], cax=cb_ax)

    return f, axarr
