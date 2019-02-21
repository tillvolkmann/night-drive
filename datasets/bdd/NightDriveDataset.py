import os
import numpy as np
import pandas as pd
from PIL import Image
from pandas import DataFrame
from torch.utils.data import Dataset


class NightDriveDataset(Dataset):

    def __init__(self, root_dir='', database="bdd_all", split=None, transform=None, augment=None, dropcls_dict=None, force_num=None):
        """ Constructor """
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.data = self._load_data(database)
        if dropcls_dict is not None:
            self.data = self._drop_classes(dropcls_dict)
        if split is not None:
            self.data = self._split_data(split)
        if force_num is not None:
            self.data_balanced = self._balance_classes(force_num)

    def __len__(self):
        """Provides the size of the dataset."""
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

    """Read Berkeley Deep Drive label json."""
    def _load_data(self, database):
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

    def _drop_classes(self, dropcls_dict):
        """Drop classes"""
        # drop samples with unwanted classes
        dropcls_dict = {'weather': ['undefined'], 'timeofday': ['undefined', 'dawn/dusk']}
        for k, v in dropcls_dict.items():
            if k in self.data.columns:
                self.data = self.data.loc[~self.data[k].isin(v)]
        return self.data.reset_index(drop=True)

    def _split_data(self, split):
        # Set of outputs for final project data sets
        print(">> Returning", split, "set.")
        # Settings for random sampling into the different sets
        n_total = self.data.shape[0]  # total size of data set
        # dictionary specifying target size (n) and class distribution (class_dist) for split
        sampler_dict = {
            'train'     : {'n': 20000, 'class_dist': {'daytime': 1.,   'dawn/dusk': 0.,     'night': 0.}},
            'train_dev' : {'n': 2.5e3, 'class_dist': {'daytime': 1.,   'dawn/dusk': 0.,     'night': 0.}},
            'test'      : {'n': 2.5e3, 'class_dist': {'daytime': 1./2, 'dawn/dusk': 0.,   'night': 1./2}},
            'valid'     : {'n': 2.5e3, 'class_dist': {'daytime': 1./2, 'dawn/dusk': 0.,   'night': 1./2}},
        }
        # create column indicating split of database into train, train-dev, dev, and test set
        # add a column to the dataframe indicating split association, initialize with all 'unassigned'
        self.data['split'] = 'unassigned'
        # shuffle data and reset index
        self.data = self.data.sample(frac=1.0, random_state=123).reset_index(drop=True)
        # stratified random sampling of indices for val set
        np.random.seed(123)
        for sp in sampler_dict.keys():  # for each subset
            for cl, fr in sampler_dict[sp]['class_dist'].items():  # for each class
                n_samples = int(sampler_dict[sp]['n'] * fr)
                if n_samples == 0:
                    continue  # efficient, but also necessary to avoid empty input to np_random_choice
                class_bool = self.data.timeofday.eq(cl) & self.data.split.eq('unassigned')
                if n_samples <= sum(class_bool):
                    idx = np.random.choice(self.data.index[class_bool], size=n_samples, replace=False)
                    self.data.loc[idx, 'split'] = sp
                else:
                    raise Exception('Insufficient records available for target sample size of split')

        # query based on split column
        return self.data.query('split == @split', inplace=False).reset_index(drop=True)

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
        self.data = self._tidy_data()
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


class DetectorDataset(NightDriveDataset):
    """"DataSet sub-class for Detector application in project Night-Dirve."""
    def __init__(self, root_dir='', database="bdd_all", split=None, transform=None, augment=None, dropcls=[None], force_num=None):
        super().__init__(root_dir, database, split, transform, augment, dropcls, force_num)
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
        self.data['bbox'] = self._condese_labels(keep_cat)
        # drop unneeded columns
        self.data.drop(columns=['weather', 'timeofday', 'scene', 'labels', 'split'], inplace=True)
        return self.data

    def _condese_labels(self, keep_cat=['all']):
        """
        Extract a condensed list of boxes of given category from a list of frame.
        Output format is [{'category: str, 'box2d': {'x1': flt, 'y1': flt, 'x2': flt, 'y2': flt}}]
        """
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


class AugGANDataset(NightDriveDataset):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    """Main implements testing."""
    from pandas.testing import assert_frame_equal

    # Set root dir
    root_dir = "/home/till/data/driving/BerkeleyDeepDrive/bdd100k"  # "/home/SharedFolder/CurrentDatasets/bdd100k"
    ds_database = 'bdd_all'
    splits = ["train", "train_dev", "valid", "test"]

    print('>>> ===========================================')
    print('>>> Performing Dataset integrity tests.')
    print('>>> ===========================================')
    # Verify that data set composition is reproducible
    for s in splits:
        # WeatherClassifierDataset
        # load data twice
        ds_A = WeatherClassifierDataset(root_dir, database=ds_database, split=s)
        ds_B = WeatherClassifierDataset(root_dir, database=ds_database, split=s)
        # evaluate hashes
        if assert_frame_equal(ds_A.data, ds_B.data):  # empty when no difference
            raise Exception('WeatherClassifierDataset data loading not reproducible for split "' + s + '".')
        # DetectorDataset
        # load data twice
        ds_A = DetectorDataset(root_dir, database=ds_database, split=s)
        ds_B = DetectorDataset(root_dir, database=ds_database, split=s)
        # evaluate hashes
        if assert_frame_equal(ds_A.data, ds_B.data):  # empty when no difference
            raise Exception('WeatherClassifierDataset data loading not reproducible for split "' + s + '".')
    else:
        print('>>> WeatherClassifierDataset data loading verified reproducible for all splits.')

    # Verify that all elements across all data splits are unique
    # WeatherClassifierDataset
    # load and merge all splits
    data = pd.DataFrame()
    for s in splits:
        ds_part = WeatherClassifierDataset(root_dir, database=ds_database, split=s)
        data = pd.concat([data, ds_part.data], axis=0)
    # Check that all elements are unique
    if data['name'].nunique() != data.shape[0]:
        raise Exception('WeatherClassifierDataset data splits contain non-unique items.')
    else:
        print('>>> WeatherClassifierDataset data splits verified to contain only unique items.')

    # All done
    print('>>> ===========================================')
    print('>>> Testing of Datasets completed successfully.')
    print('>>> ===========================================')


