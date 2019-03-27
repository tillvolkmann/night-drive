import matplotlib.pyplot as plt
import os, sys
import matplotlib.image as mpimg
import math
from random import shuffle
import pandas as pd
import numpy as np
import re


def load_bdd_json(json_path, data_root):
    """
    Load and pre-process Berkeley Deep Drive-style json file.

    :param json_path:
    :param data_path:
    :return:
    """

    # read data
    df = pd.read_json(json_path)
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


def plot_imagegrid(names, max_num_img=64, save_name=None, fig_title=None, show_fig=True, max_ncol=8, do_shuffle=True, labels=None, show_name=False, regex=None):
    """
    Displays images in a grid-like arrangement.

    :param names: list of image names or directories containing (currently only) images
    :param max_num_img: 
    :param save_name: 
    :param fig_title: 
    :param show_fig: 
    :param max_ncol: 
    :param do_shuffle: 
    :param labels: 
    :param show_name: 
    :return: 
    """

    # check whether dir is given
    if isinstance(names, list):  # multiple dirs
        if os.path.isdir(names[0]):
            im_dirs = names.copy()
            names = list()
            for im_dir in im_dirs:
                files = list(os.listdir(im_dir))
                for file in files:
                    names.append(os.path.join(im_dir, file))
    elif os.path.isdir(names):  # single dir
        im_dir = names
        names = list()
        files = list(os.listdir(im_dir))
        for file in files:
            names.append(os.path.join(im_dir, file))

    # allows keeping only images that match pattern in filter
    names_filtered = list()
    if regex is not None:
        for name in names:
            if re.search(regex, name) is not None:
                names_filtered.append(name)
    names = names_filtered

    # get number of images
    n_img = len(names)
    if n_img > max_num_img:
        n_img = min(n_img, max_num_img)

    # shuffle if requested
    if do_shuffle:
        shuffle(names)

    # get number of rows and columns
    ncol = int(np.round(n_img ** (0.5)))
    if ncol < max_ncol:
        nrow = int(math.ceil(n_img ** (0.5)))
    else:
        ncol = max_ncol
        nrow = int(math.ceil(n_img / ncol))

    # get figure size
    figh = nrow * 3
    figw = ncol * 3 * (16 / 9)
    fig, ax = plt.subplots(nrow, ncol,
                           figsize=[figw, figh], gridspec_kw={'wspace': 0.03, 'hspace': 0.03}, squeeze=True)

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
                if show_name is True:
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
    if show_fig:
        fig.show()

    # save figure to disk
    if save_name is not None:
        print("Saving figure file {}".format(save_name))
        path, file = os.path.split(save_name)
        if not os.path.exists(path):
            if __name__ == '__main__':
                os.makedirs(path)
        plt.savefig(save_name+".jpg", format='jpg', dpi=90, bbox_inches='tight')


def get_filepath(root, fname_search, notfounderror=True):
    """
    Finds the full path to fname_search within the directory tree of root.
    Throws exception if file does not exist under this root.

    :param root: root directory for search
    :param fname_search: filename to find
    :param notfounderror: Set to False if no exception to be raised when a file is not found on root (returns None instead).
    :return: full path to fname_search
    """
    for dir, _, fnames in os.walk(root):  # search through all dirs on path
        for fname in fnames:  # search through all files on path
            if fname in fname_search:
                return os.path.join(dir, fname)  # if found, return full path
    if notfounderror:
        raise Exception(
            "Couldn't load image {} from root {}".format(fname_search, root))  # if not found anywhere, raise exception
    else:
        return None


def get_filepaths(root, fnames_search, notfounderror=True):
    """
    Finds the full paths to all files in fnames_search within the directory tree of root.
    Throws exception if any one file does not exist under this root.

    :param root: root directory for search
    :param fname_search: filenames to find
    :param notfounderror: Set to False if no exception to be raised when a file is not found on root.
    :return: full paths to fnames_search
    """
    for fname_search in fnames_search:
        get_filepath(root, fname_search, notfounderror)
    print(f"Found all files to be searched for in {root}.")


# CAM code from https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py
def get_CAM_resnet18(image_path, weights_path, class_dict, n_outputs, device):
    import cv2
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.nn import functional as F
    from torchvision import models, transforms
    from torch.autograd import Variable

    net = models.resnet18(pretrained = True)
    finalconv_name = 'layer4'
    net.fc = nn.Linear(net.fc.in_features, n_outputs)
    net.load_state_dict(torch.load(weights_path)["model_state_dict"])
    net.eval()

    # hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    img_pil = Image.open(image_path)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))

    net.to(torch.device(device))
    img_variable = img_variable.to(torch.device(device))
    logit = net(img_variable)
    logit = logit.cpu()

    h_x = F.softmax(logit, dim = 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result_bgr = heatmap * 0.3 + img * 0.5

    return class_dict[idx[0]], probs[0], result_bgr


if __name__ == '__main__':
    pass


