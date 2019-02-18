import os
import numpy as np
import pandas as pd
from PIL import Image
from pandas import DataFrame
from torch.utils.data import Dataset


class NightDriveDataset(Dataset):

    def __init__(self, root_dir='', database="bdd_all", split=None, transform=None, augment=None, dropcls=[None], force_num=None):
        """ Constructor """
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.data = self._load_data(database, dropcls)
        if split is not None:
            self.data = self._split_data(split)
        if force_num is not None:
            self.data_balanced = self._balance_classes(force_num)

    def __len__(self):
        """Provides the size of the dataset."""
        return self.data.shape[0]

    def __getitem__(self, ix):
        """Provides one sample (image + target) from index."""
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

    def _load_image(self, ix):
        """Load one image from disk as PIL image."""
        img_path = self.data.name[ix]
        img = Image.open(img_path)
        return img

    def _get_class_dict(self):
        """ Get class-int mapping """
        return self.class_dict

    """Read Berkeley Deep Drive label json."""
    def _load_data(self, database, dropcls):
        # Load databases
        data = pd.DataFrame()
        if database in ['bdd_train', 'bdd_all', 'all']:
            print(">> Loading BDD training label dataset")
            # training set
            data_new = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_train.json"))
            # concat paths to images, since different for training and val set
            data_new["name"] = self.root_dir + "/images/100k/train/" + data_new["name"].astype(str)
            data = pd.concat([data, data_new], axis=0)
        if database in ['bdd_valid', 'bdd_all', 'all']:
            print(">> Loading BDD validation label dataset")
            # validation set
            data_new = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_val.json"))
            # concat paths to images, since different for training and val set
            data_new["name"] = self.root_dir + "/images/100k/val/" + data_new["name"].astype(str)
            data = pd.concat([data, data_new], axis=0)

        # Some data clean up:
        # drop unneeded columns
        data.drop(columns=['timestamp'], inplace=True)
        # extract attributes into colums
        keep_attributes = ['weather', 'scene', 'timeofday']
        for key in keep_attributes:
            data[key] = data.attributes.apply(lambda x: x[key])
        data.drop(columns=['attributes'], inplace=True)
        # extract relevant labels
        # keep_label_categories = []
        # keep_label_attributes = []
        # for key in keep_label_categories:
        #     data[key] = data.attributes.apply(lambda x: x[key])

        # drop samples with unwanted classes
        data = data.loc[~data.weather.isin(dropcls)].reset_index(drop=True)

        # query based on split column
        return data

    def _split_data(self, split):
        # Set of outputs for final project data sets
        print(">> Returning", split, "set.")
        # Settings for random sampling into the different sets
        n_total = self.data.shape[0]  # total size of data set
        # dictionary specifying target size (n) and class distribution (class_dist) for split
        sampler_dict = {
            'train'     : {'n': 37800, 'class_dist': {'daytime': 1.,   'dawn/dusk': 0.,     'night': 0.}},
            'train_dev' : {'n': 2.5e3, 'class_dist': {'daytime': 1.,   'dawn/dusk': 0.,     'night': 0.}},
            'test'      : {'n': 2.5e3, 'class_dist': {'daytime': 1./3, 'dawn/dusk': 1./3,   'night': 1./3}},
            'valid'     : {'n': 2.5e3, 'class_dist': {'daytime': 1./3, 'dawn/dusk': 1./3,   'night': 1./3}},
        }
        # error check for completeness of kws and total num samples?

        # create column indicating split of database into train, train-dev, dev, and test set
        # add a column to the dataframe indicating split association, initialize with all 'unassigned'
        self.data['split'] = pd.Series(np.array('unassigned').repeat(n_total))
        # shuffle data and reset index
        self.data = self.data.sample(frac=1.0, random_state=123).reset_index(drop=True)
        # stratified random sampling of indices for val set
        for s in sampler_dict.keys():  # for each subset
            for c in sampler_dict[s]['class_dist'].keys():  # for each class
                n_class_val = int(sampler_dict[s]['n'] * sampler_dict[s]['class_dist'][c])
                self.data['split'].loc[self.data.loc[(self.data['timeofday'] == c) & (self.data['split'] == 'unassigned')].sample(n_class_val, random_state=123, replace=False).index] = s

        # query based on split column
        return self.data.query('split == @split', inplace=False)

    # balance classes to fixed number
    def _balance_classes(self, force_num):
        if force_num is not None:
            data_balanced = pd.DataFrame()
            for weather, count in self.data.loc[:, "weather"].value_counts().to_dict().items():
                if force_num >= count:
                    orig = self.data.loc[self.data.weather == weather]
                    # sample with replacement
                    over = orig.iloc[np.random.randint(0, orig.shape[0], size = force_num - orig.shape[0])]
                    data_balanced = pd.concat([data_balanced, orig, over], axis = 0)
                else:
                    orig = self.data.loc[self.data.weather == weather].iloc[0:force_num]
                    data_balanced = pd.concat([data_balanced, orig], axis = 0)
        return data_balanced


class WeatherClassifierDataset(NightDriveDataset):
    """"DataSet sub-class for Weatehr classifier in project Night-Dirve."""
    # Why does this not work???
    #     def __init__(self, *args):
    #     super().__init__(*args)
    def __init__(self, root_dir='', database="bdd_all", split=None, transform=None, augment=None, dropcls=[None], force_num=None):
        super().__init__(root_dir, database, split, transform, augment, dropcls, force_num)
        self.class_dict = dict(zip(
            list(np.sort(self.data.weather.unique())),
            list(range(self._get_num_classes())))
        )

    def _get_target(self, ix):
        """Get one target (weather classification)."""
        target = self.data.iloc[ix]["weather"]
        return target

    def _get_class_counts(self):
        """Get target value count."""
        return self.data.weather.value_counts().to_dict()

    def _get_num_classes(self):
        """Get number of classes."""
        return len(self.data.weather.unique())


class DetectorDataset(NightDriveDataset):
    """"DataSet sub-class for Detector in project Night-Dirve."""
    def __init__(self, root_dir='', database="bdd_all", split=None, transform=None, augment=None, dropcls=[None], force_num=None):
        super().__init__(root_dir, database, split, transform, augment, dropcls, force_num)

    def _get_target(self, ix):
        """Get bounding boxes for current image."""
        pass

    def _get_num_boxes(self):
        """Get number of bounding boxes per category for current image."""
        pass

class AugGANDataset(NightDriveDataset):
    def __init__(self):
        super().__init__()
