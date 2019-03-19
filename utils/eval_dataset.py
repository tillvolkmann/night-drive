import os
import numpy as np
import pandas as pd
from PIL import Image


class EvalDataset():
    """
    split: str specifying split sub-set to return. Values include ['train', 'train_dev', 'valid', 'test', 'all']. Option
            'all' returns complete data set with split association indicated in column 'split'.
    """
    def __init__(self, root_dir, database=None):
        """ Constructor """
        #
        self.root_dir = root_dir
        self.database = database
        #
        self.list_splits = ["train_A", "train_B", "train_C", "train_dev_A", "train_dev_B", "train_dev_C", "valid", "test"]
        self.list_splits_over = ["train_A_over", "train_B_over", "train_C_over", "train_dev_A_over", "train_dev_B_over", "train_dev_C_over"]
        #
        self.data, self.image_base_path = self._load_data()  #
        #
        # self.dict_weather_classes = self._set_weather_class_dict()
        # self.dict_timeofday_classes = self._set_timeofday_class_dict()
        #
        self.list_weather = sorted(self.data.weather.unique().tolist())
        self.list_timeofday = sorted(self.data.timeofday.unique().tolist())

        # remove unneeded data
        # self.data = self._tidy_data()

    def __getitem__(self, ix):
        """Provides one sample (image only) from index and applies transforms and augments."""
        # load image and target
        img = self._load_image(ix)
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
            'train_A' : experimental bdd train data set A
            'train_B' : experimental bdd train data set B
            'train_C' : experimental bdd train data set C
        """
        # Load databases
        data = pd.DataFrame()
        if isinstance(self.database, str):
            self.database = list([self.database])
        for split in self.database:
            print(">> Loading annotations for data set '{}' ...".format(split))
            # print(type(split))
            # print(split)
            if any([x == split for x in ['bdd_train', 'bdd_all']]):            # training set
                # read data annotations json
                data_new = pd.read_json(os.path.join(self.root_dir, "bdd100k/labels/bdd100k_labels_images_train.json"))
                # concat paths to images, since different for training and val set
                image_base_path = os.path.join(self.root_dir, "images/100k/train/")
                data_new["name"] = image_base_path + "/" + data_new["name"].astype(str)
                print("A")
            elif any([x == split for x in ['bdd_valid', 'bdd_all']]):       # validation set
                # read data annotations json
                data_new = pd.read_json(os.path.join(self.root_dir, "bdd100k/labels/bdd100k_labels_images_val.json"))
                # concat paths to images, since different for training and val set
                image_base_path = os.path.join(self.root_dir, "images/100k/val/")
                data_new["name"] = image_base_path + "/" + data_new["name"].astype(str)
                print("AA")
            elif any([x == split for x in self.list_splits]):
                # read data annotations json
                filename = "bdd100k_sorted_" + split + ".json"
                data_new = pd.read_json(os.path.join(self.root_dir, "bdd100k_sorted/annotations/"+filename))
                # concat paths to images, since different for training and val set
                image_base_path = os.path.join(self.root_dir, "bdd100k_sorted/", split)
                data_new["name"] = image_base_path + "/" + data_new["name"].astype(str)
                print("AA")
            elif any([x == split for x in self.list_splits_over]):
                # read data annotations json
                filename = "bdd100k_sorted_" + split + ".json"
                data_new = pd.read_json(os.path.join(self.root_dir, "bdd100k_sorted/annotations/", filename))
                # concat paths to images, since different for training and val set
                image_base_path = os.path.join(self.root_dir, "bdd100k_sorted/", split[:-5])
                data_new["name"] = image_base_path + "/" + data_new["name"].astype(str)
                print("AA")

            data = pd.concat([data, data_new], axis=0)

        # Simplify data structure
        data['weather'] = data.attributes.apply(lambda x: x['weather'])
        data['timeofday'] = data.attributes.apply(lambda x: x['timeofday'])
        data['scene'] = data.attributes.apply(lambda x: x['scene'])
        # Remove unneeded / redundant columns
        data.drop(columns=['timestamp', 'attributes'], inplace=True)

        # Remove duplicates (if any)
        # data.drop_duplicates(subset=['name'], keep='first', inplace=True)

        # reset index
        data.reset_index(inplace=True, drop=True)

        # return full data
        return data, image_base_path

    def get_crosstab_timeofdayxweather(self):
        crosstab = pd.crosstab(self.data['timeofday'], self.data['weather'])
        crosstab = crosstab.reindex(sorted(crosstab.columns), axis=1)
        return crosstab

    def _drop_duplicates(self):
        # Remove duplicates (if any)
        self.data = self.data.drop_duplicates(subset=['name'], keep='first', inplace=True)

    def _drop_classes(self, dropclass_dict):
        """Drop classes"""
        # drop samples with unwanted classes
        for k, v in self.dropclass_dict.items():
            if k in self.data.columns:
                self.data = self.data.loc[~self.data[k].isin(v)]
        return self.data.reset_index(drop=True)

    def _merge_classes(self, mergeclass_dict):
        """Merge classes"""
        # merge classes
        for kk, vv in self.mergeclass_dict.items():  # for each variable key kk
            if kk in self.data.columns:
                for k, v in vv.items():  # for each subdict of 'newclass': [list of old classes]
                    self.data[kk].replace(v, k, inplace=True)
        return self.data.reset_index(drop=True)

    def _get_num_weather_classes(self):
        """Get number of classes."""
        return len(self.data.weather.unique())

    def _get_weather_class_dict(self):
        """ Get class-int mapping """
        return self.weather_class_dict

    def _get_num_timeofday_classes(self):
        """Get number of classes."""
        return len(self.data.timeofday.unique())

    def _get_timeofday_class_dict(self):
        """ Get class-int mapping """
        return self.timeofday_class_dict

    def _set_timeofday_class_dict(self):
        """ Get class-int mapping """
        self.timeofday_class_dict = dict(zip(
            list(np.sort(self.data.timeofday.unique())),
            list(range(self._get_num_timeofday_classes())))
        )
        return self.timeofday_class_dict

    def _set_weather_class_dict(self):
        """ Get class-int mapping """
        self.weather_class_dict = dict(zip(
            list(np.sort(self.data.weather.unique())),
            list(range(self._get_num_weather_classes())))
        )
        return self.weather_class_dict




    # Old detector functions, need to adjust:

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

    def _get_num_boxes_per_frame(self):
        """Get number of bounding boxes per category for each frame as an array."""
        return self.data.bbox.apply(lambda x: len(x)).values


if __name__ == '__main__':
    """Main implements testing."""
    from pandas.testing import assert_frame_equal

