import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BDDWeatherDataset(Dataset):

    """ Constructor """
    def __init__(self, bdd_json_path, bdd_image_path, transform = None, augment = None, dropcls = [None]):
        self.bdd_json_path = bdd_json_path
        self.bdd_image_path = bdd_image_path
        self.transform = transform
        self.augment = augment
        self.data = self._load_data(dropcls)
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
    def _load_data(self, dropcls):
        # load annotation file
        data = pd.read_json(self.bdd_json_path)
        print(f">> Loading BDD data from {self.bdd_json_path}...")
        # complete path, since only images names in label file
        sep = "" if self.bdd_image_path[-1] == "/" else "/"
        data["name"] = self.bdd_image_path + sep + data["name"].astype(str)
        # drop other attributes
        data["weather"] = data.attributes.apply(lambda x: x["weather"])
        # drop other columns
        data = data.drop(columns = ["labels", "timestamp", "attributes"])
        # drop samples with unwanted classes
        data = data.loc[~data.weather.isin(dropcls)]
        data = data.sample(frac = 1, random_state = 123).reset_index(drop = True)
        return data
