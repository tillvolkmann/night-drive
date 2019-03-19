import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import math
from random import shuffle
import pandas as pd
import numpy as np

def load_bdd_json(json_path, data_root):
    """
    Load and pre-process Berkeley Deep Drive-style json file.

    :param json_path:
    :param data_path:
    :return:
    """

    # read data
    df = pd.read_json(path_newjson)
    # Simplify structure for easier referencing
    df['weather'] = df.attributes.apply(lambda x: x['weather'])
    df['timeofday'] = df.attributes.apply(lambda x: x['timeofday'])
    df['scene'] = df.attributes.apply(lambda x: x['scene'])
    # Remove unneeded / redundant columns
    df.drop(columns=['timestamp', 'attributes'], inplace=True)
    # Remove duplicates (if any)
    # df.drop_duplicates(subset=['name'], keep='first', inplace=True)
    # reset index
    df.reset_index(inplace=True, drop=True)
    return df


def load_main_json(json_path, data_path):
    """
    Load and pre-process main summary project json file.

    :param json_path:
    :param data_path:
    :return:
    """
    pass


def plot_imagegrid(names, max_num_img=64, save_name=None, fig_title=None, show=True, max_ncol=8, do_shuffle=True, labels=None, print_name=False):
    """
    Displays images in a grid-like arrangement.


    """

    n_img = len(names)
    if n_img > max_num_img:
        n_img = min(n_img, max_num_img)

    if do_shuffle:
        shuffle(names)

    # ncol = int(n_img ** (0.5))
    ncol = int(np.round(n_img ** (0.5)))
    if ncol < max_ncol:
        nrow = int(math.ceil(n_img ** (0.5)))
    else:
        ncol = max_ncol
        nrow = int(math.ceil(n_img / ncol))

    figh = nrow * 3
    figw = ncol * 3 * (16 / 9)
    fig, ax = plt.subplots(nrow, ncol,
                           figsize=[figw, figh], gridspec_kw={'wspace':0.03, 'hspace':0.03}, squeeze=True)
    
    # Plot images
    # for c in range(n_img):
    #     row = c // ncol
    #     col = c % ncol
    #     # load image
    #     img = mpimg.imread(names[c])
    #     # plot
    #     imgplot = ax[row, col].imshow(img, interpolation="none")
    #     #
    #     xytext_base = (2,3)
    #     if print_name is True:
    #         name = os.path.basename(names[c])
    #         ax[row, col].annotate(name, (0,0), xytext=xytext_base, xycoords='axes points', fontsize=12, color='white')
    #         xytext_base = (2,17)
    #     if labels is not None:
    #         ax[row, col].annotate(str(labels[c]), (0,0), xytext=xytext_base, xycoords='axes points', fontsize=12, color='white')
    #     # style
    #     ax[row, col].set_xticks([])
    #     ax[row, col].set_yticks([])

    # Plot images
    c = 0
    for row in range(nrow):
        for col in range(ncol):
            
            if c < n_img:
                # load image
                img = mpimg.imread(names[c])
                # plot
                imgplot = ax[row, col].imshow(img, interpolation="none")
                #
                xytext_base = (2,3)
                if print_name is True:
                    name = os.path.basename(names[c])
                    ax[row, col].annotate(name, (0,0), xytext=xytext_base, xycoords='axes points', fontsize=12, color='white')
                    xytext_base = (2,17)
                if labels is not None:
                    ax[row, col].annotate(str(labels[c]), (0,0), xytext=xytext_base, xycoords='axes points', fontsize=12, color='white')
                # style
                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([])
                # ax[row, col].set_frame_off
            else:
                # style
                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([])
                # ax[row, col].set_axis_off

            c += 1

    # add title to figure
    if fig_title is not None:
        ax[0, math.floor(ncol / 2)].set_title(fig_title, fontsize=20, fontweight='bold')

    # show figure
    if show:
        fig.show()

    # save figure to disk
    if save_name is not None:
        print("Saving figure file {}".format(save_name))
        plt.savefig(save_name+".png", format='png', dpi=90, bbox_inches='tight')
