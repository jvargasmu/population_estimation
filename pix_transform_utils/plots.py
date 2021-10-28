import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 


def plot_result(source_map, predicted_target_img, predicted_target_img_adj, validation_map, fig_size=(16, 4)):
    # cmap = "Spectral"


    vmin = np.min([0.0, np.min(source_map), np.min(predicted_target_img), np.min(predicted_target_img_adj), np.min(validation_map)])
    allmaxs = [np.max(source_map), np.max(predicted_target_img), np.max(predicted_target_img_adj), np.max(validation_map)]
    maxindex = np.argmax(allmaxs)
    vmax = allmaxs[maxindex]

    eps = 0.05

    f, axarr = plt.subplots(2, 2, figsize=fig_size, num=102) 

    a =[]
    a.append( axarr[0,0].matshow(source_map+eps, vmin=eps, vmax=vmax, norm=LogNorm(vmin=eps, vmax=vmax)) )
    a.append( axarr[1,0].matshow(predicted_target_img+eps, vmin=eps, vmax=vmax, norm=LogNorm(vmin=eps, vmax=vmax)) )
    a.append( axarr[0,1].matshow(validation_map+eps, vmin=eps, vmax=vmax, norm=LogNorm(vmin=eps, vmax=vmax)) )
    a.append( axarr[1,1].matshow(predicted_target_img_adj+eps, vmin=eps, vmax=vmax, norm=LogNorm(vmin=eps, vmax=vmax)) )

    titles = ['Source (Admin2)', 'Predicted Target',  'Target (Admin4)', 'Adj. Predicted Target',]

    for i, ax in enumerate(axarr.flatten()):
        ax.set_axis_off()
        ax.set_title(titles[i])

    f.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.4, hspace=0.1)
    cb_ax = f.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = f.colorbar(a[maxindex], cax=cb_ax)

    return f, axarr
