import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BDDWeatherDataset(Dataset):

    """ Constructor """
    def __init__(self, root_dir, transform = None, augment = None, split = "bddtrain", dropcls = [None], force_num = None):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.data = self._load_data(split, dropcls, force_num)
        self.class_dict = dict(zip(list(np.sort(self.data.weather.unique())), list(range(self._get_num_classes()))))

    """ Provides the size of the dataset """
    def __len__(self):
        return self.data.shape[0]

    """ Provides one sample (image + target) from index """
    def __getitem__(self, ix):
        # load image and target
        img = self._load_image(ix)
        target = self.class_dict[self._get_target(ix)]
        # transform images
        if self.transform is not None:
            img = self.transform(img)
        # augment image
        if self.augment is not None:
            img = self.augment(img)
        return img, target

    """ Load one image from disk as PIL image """
    def _load_image(self, ix):
        img_path = self.data.name[ix]
        img = Image.open(img_path)
        return img

    """ Get one target (weather classification) """
    def _get_target(self, ix):
        target = self.data.iloc[ix]["weather"]
        return target

    """ Get target value count """
    def _get_class_counts(self):
        return self.data.weather.value_counts().to_dict()

    """ Get number of classes """
    def _get_num_classes(self):
        return len(self.data.weather.unique())

    """ Get class-int mapping """
    def _get_class_dict(self):
        return self.class_dict

    """ Read Berkeley Deep Drive label json. """
    def _load_data(self, split, dropcls, force_num):
        if split == "bddtrain":
            print(">> Using original BDD training set")
            # training set
            data = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_train.json"))
            # concat path, since different for training and val set
            data["name"] = self.root_dir + "/images/100k/train/" + data["name"].astype(str)
        elif split == "bddvalid":
            print(">> Using original BDD validation set")
            # validation set
            data = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_val.json"))
            # concat path, since different for training and val set
            data["name"] = self.root_dir + "/images/100k/val/" + data["name"].astype(str)
        else:
            print(">> Mixing original BDD training and validation sets")
            # training set
            data_train = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_train.json"))
            # concat path, since different for training and val set
            data_train["name"] = self.root_dir + "/images/100k/train/" + data_train["name"].astype(str)
            # validation set
            data_val = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_val.json"))
            # concat path, since different for training and val set
            data_val["name"] = self.root_dir + "/images/100k/val/" + data_val["name"].astype(str)
            data = pd.concat([data_train, data_val], axis = 0)
        # drop other attributes
        data["weather"] = data.attributes.apply(lambda x: x["weather"])
        # drop other columns
        data = data.drop(columns = ["labels", "timestamp", "attributes"])
        # drop samples with unwanted classes
        data = data.loc[~data.weather.isin(dropcls)]
        data = data.sample(frac = 1, random_state = 123).reset_index(drop = True)
        # balance classes to fixed number
        if force_num is not None:
            data_balanced = pd.DataFrame()
            for weather, count in data.loc[:, "weather"].value_counts().to_dict().items():
                if force_num >= count:
                    orig = data.loc[data.weather == weather]
                    # sample with replacement
                    over = orig.iloc[np.random.randint(0, orig.shape[0], size = force_num - orig.shape[0])]
                    data_balanced = pd.concat([data_balanced, orig, over], axis = 0)
                else:
                    orig = data.loc[data.weather == weather].iloc[0:force_num]
                    data_balanced = pd.concat([data_balanced, orig], axis = 0)
            data = data_balanced.reset_index(drop = True)
        return data
