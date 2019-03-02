import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    split: str specifying split sub-set to return. Values include ['train', 'train_dev', 'valid', 'test', 'all']. Option
            'all' returns complete data set with split association indicated in column 'split'.
    """
    def __init__(self, config, split=None, transform=None, augment=None):
        """ Constructor """
        self.config = config
        self.transform = transform
        self.augment = augment
        self.data = self._load_data()
        if self.config.dropclass_dict is not None:
            self.data = self._drop_classes()
        if self.config.mergeclass_dict is not None:
            self.data = self._merge_classes()
        if split is not None:
            self.data = self._split_data(split)

    def __len__(self):
        """Provides the size (number of images) of the dataset."""
        return self.data.shape[0]

    def __getitem__(self, ix):
        """Provides one sample (image only) from index and applies transforms and augments."""
        # load image and target
        img = self._load_image(ix)
        # transform images
        if self.transform is not None:
            img = self.transform(img)
        # augment image
        if self.augment is not None:
            img = self.augment(img)
        return img

    def _load_image(self, ix):
        """Load one image from disk as PIL image."""
        img_path = self.data.name[ix]
        img = Image.open(img_path)
        return img

    def _load_data(self):
        """Read Berkeley Deep Drive label json.

            'bdd_train': original bdd train set
            'bdd_valid': original bdd valid set
            'bdd_all': original bdd train & valid sets
            'bdd_train_expA' : experimental bdd train data set A
            'bdd_train_expB' : experimental bdd train data set B
            'bdd_train_expC' : experimental bdd train data set C
        """
        # Load databases
        data = pd.DataFrame()
        if self.config.database in ['bdd_train', 'bdd_all', 'all']:
            print(">> Loading BDD training label dataset")
            # training set
            data_new = pd.read_json(os.path.join(self.config.root_dir, "labels/bdd100k_labels_images_train.json"))
            # concat paths to images, since different for training and val set
            data_new["name"] = self.config.root_dir + "/images/100k/train/" + data_new["name"].astype(str)
            data = pd.concat([data, data_new], axis=0)
        if self.config.database in ['bdd_valid', 'bdd_all', 'all']:
            print(">> Loading BDD validation label dataset")
            # validation set
            data_new = pd.read_json(os.path.join(self.config.root_dir, "labels/bdd100k_labels_images_val.json"))
            # concat paths to images, since different for training and val set
            data_new["name"] = self.config.root_dir + "/images/100k/val/" + data_new["name"].astype(str)
            data = pd.concat([data, data_new], axis=0)
        # Simplify data structure
        data['weather'] = data.attributes.apply(lambda x: x['weather'])
        data['timeofday'] = data.attributes.apply(lambda x: x['timeofday'])
        data['scene'] = data.attributes.apply(lambda x: x['scene'])
        # Remove unneeded / redundant columns
        data.drop(columns=['timestamp', 'attributes'], inplace=True)
        # Remove duplicates (if any)
        data.drop_duplicates(subset=['name'], keep='first', inplace=True)
        # reset index
        data.reset_index(inplace=True, drop=True)
        # return full data
        return data

    def _drop_classes(self):
        """Drop classes"""
        # drop samples with unwanted classes
        for k, v in self.config.dropclass_dict.items():
            if k in self.data.columns:
                self.data = self.data.loc[~self.data[k].isin(v)]
        return self.data.reset_index(drop=True)

    def _merge_classes(self):
        """Merge classes"""
        # merge classes
        for kk, vv in self.config.mergeclass_dict.items():  # for each variable key kk
            if kk in self.data.columns:
                for k, v in vv.items():  # for each subdict of 'newclass': [list of old classes]
                    self.data[kk].replace(v, k, inplace=True)
        return self.data.reset_index(drop=True)


class WeatherClassifierDataset(BaseDataset):
    """"DataSet sub-class for Weatehr classifier in project Night-Drive."""
    # Why does this not work???
    #     def __init__(self, *args):
    #     super().__init__(*args)
    def __init__(self, config, split=None, transform=None, augment=None):
        super().__init__(config, split, transform, augment)
        # remove unneeded data
        self.data = self._tidy_data()
        # get a dict of weather classes
        self.class_dict = dict(zip(
            list(np.sort(self.data.weather.unique())),
            list(range(self._get_num_classes())))
        )

    def __getitem__(self, ix):
        """Provides one sample (image & target) from index and applies transforms and augments."""
        img = super().__getitem__(ix)
        target = self.class_dict[self._get_target(ix)]
        return img, target

    def _tidy_data(self):
        # drop unneeded columns
        self.data.drop(columns=['timeofday', 'scene', 'labels', 'split'], inplace=True)
        return self.data

    def _get_target(self, ix):
        """Get one target (weather classification)."""
        return self.data.weather.iloc[ix]

    def _get_class_counts(self):
        """Get target value count."""
        return self.data.weather.value_counts().to_dict()

    def _get_num_classes(self):
        """Get number of classes."""
        return len(self.data.weather.unique())

    def _get_class_dict(self):
        """ Get class-int mapping """
        return self.class_dict


class DetectorDataset(BaseDataset):
    """"DataSet sub-class for Detector application in project Night-Dirve."""
    def __init__(self, config, split=None, transform=None, augment=None):
        super().__init__(config, split, transform, augment)
        # remove unneeded data
        self.data = self._tidy_data()

    def __getitem__(self, ix):
        """Provides one sample (image & target) from index and applies transforms and augments."""
        img = super().__getitem__(ix)
        target = self._get_target(ix)
        return img, target

    def _tidy_data(self):
        """Tidy up data for detector training."""
        # extract releveant label data
        keep_cat = ['pedestrian', 'traffic sign']  # specify categories to keep
        self.data['bbox'] = self._condense_labels(keep_cat)
        # drop unneeded columns
        self.data.drop(columns=['weather', 'timeofday', 'scene', 'labels', 'split'], inplace=True)
        return self.data

    def _condense_labels(self, keep_cat=None):
        """
        Extract a condensed list of boxes of given category from a list of frame.
        Output format is [{'category: str, 'box2d': {'x1': flt, 'y1': flt, 'x2': flt, 'y2': flt}}]
        """
        # default value for keep_cat (mutable object), keep all categories
        if keep_cat is None:
            keep_cat = ['all']
        # extract bounding boxes
        bbox = np.empty((len(self.data), 0)).tolist()  # instantiate column of empty lists
        for ixf, frame in enumerate(self.data.labels):
            for label in frame:
                if label['category'] in keep_cat or [keep_cat] == ['all']:
                    bbox[ixf].append({'category': label['category'], 'box2d': label['box2d']})
        return bbox

    def _get_target(self, ix):
        """Get dict of all bounding boxes for current image."""
        return self.data.bbox.iloc[ix]

    def _get_num_boxes_per_frame(self):
        """Get number of bounding boxes per category for each frame as an array."""
        return self.data.bbox.apply(lambda x: len(x)).values


class AugGANDataset(BaseDataset):
    def __init__(self):
        super().__init__()


class GetConfig():
    """
    Loads config file for classifier and detector training.
    Available files:
        ~ 'config_bdd_setA.json' : 100% day, 0% night
        ~ 'config_bdd_setB.json' : 75% day, 25% night
        ~ 'config_bdd_setC.json' : 50% day, 50% night
    """
    def __init__(self, filename):
        import json
        import os
        # Load config file
        try:
            with open(filename, 'r') as f:
                content = json.load(f)
        except FileNotFoundError:   # try to load cfg file from config directory, works when cwd somewhere within night drive
            cfg_path = os.path.join(os.getcwd()[:os.getcwd().find("night-drive/") + 12], "config/")
            cfg_path = os.path.join(cfg_path, os.path.basename(filename))
            with open(cfg_path, 'r') as f:
                content = json.load(f)

        # unpack content to be accessible as attributes
        self.__dict__ = content

        # adjust root dir by user (convenience for Till and Christoph)
        if "home/till/" in os.getcwd():
            self.root_dir = "/home/till/data/driving/BerkeleyDeepDrive/bdd100k"
        elif "home/SharedFolder/" in os.getcwd():
            self.root_dir = "/home/SharedFolder/CurrentDatasets/bdd100k"


if __name__ == '__main__':
    """Main implements testing."""
    from pandas.testing import assert_frame_equal

    splits = ["train", "train_dev", "valid", "test"]

    print('>>> ===========================================')
    print('>>> Performing Dataset integrity tests.')
    print('>>> ===========================================')
    # Verify that data set composition is reproducible for a given split and set (test only perfomed for set A)
    config_file = '../../config_bdd_setA.json'
    config = GetConfig(config_file)  # see docstring for info on available config files
    for spl in splits:
        # WeatherClassifierDataset
        # load data twice
        ds_A = WeatherClassifierDataset(config, split=spl)
        ds_B = WeatherClassifierDataset(config, split=spl)
        # evaluate hashes
        if assert_frame_equal(ds_A.data, ds_B.data):  # empty when no difference
            raise Exception('WeatherClassifierDataset data loading not reproducible for split "' + spl + '".')
        # DetectorDataset
        # load data twice
        ds_A = DetectorDataset(config, split=spl)
        ds_B = DetectorDataset(config, split=spl)
        # evaluate hashes
        if assert_frame_equal(ds_A.data, ds_B.data):  # empty when no difference
            raise Exception('DetectorDataset data loading not reproducible for split "' + spl + '".')
    else:
        print('>>> WeatherClassifierDataset data loading verified reproducible for all splits.')

    # Verify that test and validation is constant across sets A,B,C
    for spl in ['valid', 'test']:
        # load data twice using WeatherClassifierDataset
        config_file = '../../config_bdd_setA.json'
        config = GetConfig(config_file)  # see docstring for info on available config files
        ds_A = WeatherClassifierDataset(config, split=spl)
        config_file = '../../config_bdd_setB.json'
        config = GetConfig(config_file)  # see docstring for info on available config files
        ds_B = WeatherClassifierDataset(config, split=spl)
        config_file = '../../config_bdd_setC.json'
        config = GetConfig(config_file)  # see docstring for info on available config files
        ds_C = WeatherClassifierDataset(config, split=spl)
        # validate consistency of validation sets across sets A, B, C
        assert ds_A.data.name.isin(ds_B.data.name).value_counts().all(), \
            'The test set is not consistent among sets A and B.'
        assert ds_A.data.name.isin(ds_C.data.name).value_counts().all(), \
            'The test set is not consistent among sets A and C.'

    # Verify that all elements across all data splits are unique
    # WeatherClassifierDataset
    # load and merge all splits
    name = pd.Series()
    for spl in splits:
        data_part = WeatherClassifierDataset(config, split=spl)
        name_part = data_part['name'].unique()  # this is to eliminate oversampled duplicates
        name = pd.concat([name, name_part])
    # Check that all elements are unique, i.e. no leaking of information
    if name.nunique() != name.size:
        raise Exception('WeatherClassifierDataset data splits contain non-unique items.')
    else:
        print('>>> WeatherClassifierDataset data splits verified to contain only unique items.')

    # All done
    print('>>> ===========================================')
    print('>>> Testing of Datasets completed successfully.')
    print('>>> ===========================================')


