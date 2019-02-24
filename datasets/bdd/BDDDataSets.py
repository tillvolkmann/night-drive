import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class NightDriveDataset(Dataset):

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

    def _load_data(self):
        """Read Berkeley Deep Drive label json."""
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

    def _split_data(self, split):
        """Split data set."""
        # cross-tabulation of available samples in space time x weather
        cross_total = pd.crosstab(self.data['timeofday'], self.data['weather'])
        cross_total = cross_total.reindex(sorted(cross_total.columns), axis=1)  # columns need to be in same order as sampler_table

        # Get some helpers
        _splits = ['test', 'valid', 'train_dev', 'train']  # self.config.sampler_dict.keys()
        _timeofday_classes = self.data.timeofday.unique()
        _weather_classes = np.sort(self.data.weather.unique())
        _num_weather_classes = len(_weather_classes)
        _num_timeofday_classes = len(_weather_classes)
        _dist_weather_classes = cross_total.div(cross_total.sum(axis=1), axis=0)

        # Initialize empty sampler table
        iterables = [_splits, _timeofday_classes]
        index = pd.MultiIndex.from_product(iterables, names=['split', 'timeofday'])
        sampler_table = pd.DataFrame(np.zeros([8, len(_weather_classes)]), index=index, columns=_weather_classes)
        sampler_table = sampler_table.reindex(sorted(sampler_table.columns), axis=1)
        over_table = sampler_table.copy()

        # First, we need to process all ocurrences of min_thr
        for s in _splits:
            if self.config.sampler_dict[s]['class_min'] is not None:
                sampler_table.loc[s] = self.config.sampler_dict[s]['class_min']
        cross_avail = cross_total - sampler_table.groupby(level='timeofday').sum()
        assert ~(cross_avail < 0).any().any(), 'Error, insufficient samples available to fullfil request.'

        # Second, case 'none' i.e. no balancing; note this case has priority over under and over
        for s in _splits:
            if self.config.sampler_dict[s]['balancing'] == 'none':
                for t, f in self.config.sampler_dict[s]['class_dist'].items():  # for each timeofday
                    if f > 0.0:  # note this will intentionally exclude dusk/dawn
                        _wanted = self.config.sampler_dict[s]['n'] * f * _dist_weather_classes.loc[t]
                        # correct for min_thr
                        _ = _wanted - sampler_table.loc[s, t]
                        __ = (sum(_wanted) + _.where(_ < 0).fillna(0).sum()) / sum(_wanted)
                        _wanted = _wanted * __
                        sampler_table.loc[s, t] = np.maximum(sampler_table.loc[s, t], _wanted)  # maximum on min_thr and balanced
        cross_avail = cross_total - sampler_table.groupby(level='timeofday').sum()
        assert ~(cross_avail < 0).any().any(), 'Error, insufficient samples available to fullfil request.'

        # Third, we need to process all cases of undersampling ('under'); note this case has priority over 'over'
        for s in _splits:
            if self.config.sampler_dict[s]['balancing'] == 'under':
                for t, f in self.config.sampler_dict[s]['class_dist'].items():  # for each timeofday
                    if f > 0.0:  # note this will intentionally exclude dusk/dawn
                        sampler_table.loc[s, t] = np.maximum(sampler_table.loc[s, t],
                                                             self.config.sampler_dict[s]['n'] * f * np.ones(
                                                                 _num_weather_classes) / _num_weather_classes)  # maximum on min_thr and balanced
        cross_avail = cross_total - sampler_table.groupby(level='timeofday').sum()
        assert ~(cross_avail < 0).any().any(), 'Error, insufficient samples available to fullfil request.'

        # Fourth, we can process the cases of oversampling
        # We over-sample across all splits that want oversampling in proportion to their n
        # Therefore, we need to know the
        n_total_over = sum(list({self.config.sampler_dict[k]['n'] for k in self.config.sampler_dict if self.config.sampler_dict[k]['balancing'] == 'over'}))
        for s in _splits:
            if self.config.sampler_dict[s]['balancing'] == 'over':
                f_total_over = self.config.sampler_dict[s][ 'n'] / n_total_over  # fraction of total remaing samples per class going into this split
                for t, f in self.config.sampler_dict[s]['class_dist'].items():  # for each timeofday
                    if f > 0.0:
                        # get max number of remaining samples available (for each weather condition)
                        _avail = cross_avail.loc[t]  # ! plus those that this split aleady has from min_thr, if any
                        # get the assigned fraction of them for this split
                        _assigned = _avail * f_total_over
                        _wanted = self.config.sampler_dict[s]['n'] * f * np.ones(_num_weather_classes) / _num_weather_classes
                        _given = np.minimum(_assigned, _wanted)
                        sampler_table.loc[s, t] = np.maximum(sampler_table.loc[s, t], _given)  # maximum on min_thr and balanced
                        # store number of samples to be oversampled per class
                        over_table.loc[s, t] = _wanted - sampler_table.loc[s, t]
        cross_avail = cross_total - sampler_table.groupby(level='timeofday').sum()
        print(cross_avail.astype('int32'))
        print(sampler_table.astype('int32'))
        assert ~(cross_avail < 0).any().any(), 'Error, insufficient samples available to fullfil request.'

        # cast to int
        sampler_table = sampler_table.astype('int32')
        over_table = over_table.astype('int32')

        # print some summary stats
        sampler_table_show = sampler_table.copy()
        sampler_table_show['total'] = sampler_table_show.sum(axis=1)
        print('\nMulti-variate distribution of unique original samples for each split:\n')
        print(sampler_table_show.reindex(sorted(sampler_table_show.index), axis=0))
        sampler_table_show = sampler_table_show.groupby(level='split').sum()
        print('\nUnique sample distribution grouped by split:\n')
        print(sampler_table_show.reindex(sorted(sampler_table_show.index), axis=0))
        print('\nUnique samples not used:\n')
        print(cross_avail.reindex(sorted(cross_avail.index), axis=0).astype('int32'))
        print('\nMulti-variate distribution of oversampling samples across splits:\n')
        over_table_show = over_table.copy()
        over_table_show['total'] = over_table_show.sum(axis=1)
        print(over_table_show.reindex(sorted(over_table_show.index), axis=0))
        #
        print('\nCross tabulation of original samples timeofday x weather:\n')
        cross_total_show = cross_total.copy()
        cross_total_show['total'] = cross_total_show.sum(axis=1)
        print(cross_total_show)

        # add a column to the dataframe indicating split association (train, train-dev, dev, and test set)
        self.data['split'] = 'unassigned'  # initialize with all 'unassigned'
        # shuffle data and reset index
        self.data = self.data.sample(frac=1.0, random_state=123).reset_index(drop=True)

        # stratified random sampling of indices for split sets based on sampler_table; note that sequential processing
        # split sets up to the selected split is necessary to obtain reproducible results
        np.random.seed(123)
        data_out = pd.DataFrame()
        for sp in _splits:   # sampler_table.index.get_level_values(level='split').unique():  # for each split set; note order matters here if reproducibility is needed, i.e. val and test need to go first
            for td in sampler_table.index.get_level_values(level='timeofday').unique():  # for each timeofday
                for we in sampler_table.columns:  # for each weather condition
                    n_samples = sampler_table.loc[(sp,td), we]
                    class_bool = self.data.split.eq('unassigned') & self.data.timeofday.eq(td) & self.data.weather.eq(we)
                    assert n_samples <= sum(class_bool), \
                        "Insufficient records ({}) available for target size ({}): split='{}', timeofday='{}', weather={}" \
                            .format(sum(class_bool), n_samples, sp, td, we)
                    idx = np.random.choice(self.data.index[class_bool].values, size=n_samples, replace=False)
                    self.data.loc[idx, 'split'] = sp
                    # we can terminate this process as soon as we are done with the requested class
                    if sp == split:
                        # query split set based on split column
                        n_over = over_table.loc[(sp, td), we]
                        if n_over > 0:
                            idx = np.hstack((idx, np.random.choice(idx, size=n_over, replace=True)))
                        data_out = pd.concat([data_out, self.data.loc[idx]], axis=0)
            # we can terminate the assignment process as soon as we are done with the requested split
            if sp == split:
                return data_out.sample(frac=1.0, replace=False, random_state=123).reset_index(drop=True)
        # in case no split was requested, we return the orignial data with the added column indicating split assignment
        return self.data.sample(frac=1.0, replace=False, random_state=123).reset_index(drop=True)


class WeatherClassifierDataset(NightDriveDataset):
    """"DataSet sub-class for Weatehr classifier in project Night-Dirve."""
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


class DetectorDataset(NightDriveDataset):
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


class AugGANDataset(NightDriveDataset):
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
        # Load config file
        import json
        with open(filename, 'r') as f:
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

    # get config
    config_file = '../../config_bdd_setA.json'
    config = GetConfig(config_file)  # see docstring for info on available config files

    # Spec splits to load
    splits = ["train", "train_dev", "valid", "test"]

    print('>>> ===========================================')
    print('>>> Performing Dataset integrity tests.')
    print('>>> ===========================================')
    # Verify that data set composition is reproducible
    for s in splits:
        # WeatherClassifierDataset
        # load data twice
        ds_A = WeatherClassifierDataset(config, split=s)
        ds_B = WeatherClassifierDataset(config, split=s)
        # evaluate hashes
        if assert_frame_equal(ds_A.data, ds_B.data):  # empty when no difference
            raise Exception('WeatherClassifierDataset data loading not reproducible for split "' + s + '".')
        # DetectorDataset
        # load data twice
        ds_A = DetectorDataset(config, split=s)
        ds_B = DetectorDataset(config, split=s)
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
        ds_part = WeatherClassifierDataset(config, split=s)
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


