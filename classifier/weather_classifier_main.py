## Night-drive condition classifier main
# ================================================
# Import modules
# ================================================
import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pretrainedmodels.models import resnet50
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image


# ================================================
# Config
# ================================================
# path to berkley deep drive main folder (assume that the basic file structure has not changed)
dir_bdd = "/home/till/data/driving/BerkeleyDeepDrive/" # arg for constructor
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ================================================
# Read label data set
# ================================================
def bdd_read_labels(main_dir):
    """Read Berkley Deep Drive label json. I found it is fater to just work with the json file, vs panda."""
    # test set
    _path_labels_train_bdd = 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
    labels_train_bdd = pd.read_json(os.path.join(main_dir, _path_labels_train_bdd))
    # concat path, since different for test and val set
    # ! Note actually that there is also a separate test set, though no separate label /annotations file
    _path_image_train = 'bdd100k_images/bdd100k/images/100k/train/'
    labels_train_bdd['name'] = main_dir + _path_image_train + labels_train_bdd['name'].astype(str)

    # validation set
    _path_labels_val_bdd = 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
    labels_val_bdd = pd.read_json(os.path.join(main_dir, _path_labels_val_bdd))
    # concat path, since different for test and val set
    _path_image_val = 'bdd100k_images/bdd100k/images/100k/val/'
    labels_val_bdd['name'] = main_dir + _path_image_val + labels_val_bdd['name'].astype(str)

    return labels_train_bdd, labels_val_bdd

# read label data
df_train_bdd, df_val_bdd = bdd_read_labels(dir_bdd)


# ================================================
# View an image (test)
# ================================================
plt.imshow(PIL.Image.open(df_train_bdd.name[0]))
plt.show()

# ================================================
# data set class for classifier (pandas version)
# ================================================
class WeatherClassifierDataset(Dataset):

    def __init__(self, meta, transform=None, augment=None):
        super().__init__()
        self.meta = meta
        self.transform = transform
        self.augment = augment

    def __len__(self):
        """provides the size of the dataset"""
        return self.meta.shape[0]

    def __getitem__(self, ix):
        # load image and target class
        im = self._load_image(ix)
        cl = self._load_target(ix)

        # transform images
        if self.transform is not None:
            im = self.transform(im)

        # augment image
        if self.augment is not None:
            im = self.augment(im)

        return im, cl

    def _load_image(self, ix):
        """Load one image from disk as PIL image"""
        _path_labels_train_bdd = 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
        _impath = self.meta.name[ix]
        _im =  PIL.Image.open(_impath)
        return _im

    def _load_target(self, ix):
        """Load one label (weather classification)"""
        _cl = self.meta.attributes.iloc[ix]['weather']
        return _cl

    def collate_func(self, batch):
        pass

# Instantiate the WeatherClassifierDatasets
ds_train = WeatherClassifierDataset(df_train_bdd)
# test
print(ds_train[1])

