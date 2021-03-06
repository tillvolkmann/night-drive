import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class BDDWeatherDataset(Dataset):

    """ Constructor """
    def __init__(self, bdd_json_path, bdd_image_path, transform = None, drop_cls = [None], force_num = None, hold_out_frac = None):
        self.bdd_json_path = bdd_json_path
        self.bdd_image_path = bdd_image_path
        self.transform = transform
        self.data, self.data_idx, self.holdout_idx = self._load_data(drop_cls, force_num, hold_out_frac)
        self.class_dict = dict(zip(list(np.sort(self.data.weather.unique())), list(range(self._get_num_classes()))))

    """ Provides the size of the dataset """
    def __len__(self):
        return self.data.shape[0]

    """ Provides one sample (image + target) from index """
    def __getitem__(self, ix):
        # load image and target
        img = self._load_image(ix)
        target = self.class_dict[self._get_target(ix)]
        # transform image
        if self.transform is not None:
            transformed = self.transform(image = img)
            img = transformed["image"]
        return (img, target, self.data.iloc[ix]["name"])

    """ Load one image from disk as PIL image """
    def _load_image(self, ix):
        img_path = self.data.iloc[ix]["name"]
        # Read an image with OpenCV
        img = cv2.imread(img_path)
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    """ Get data indices (debug) """
    def _get_idx(self):
        return self.data_idx, self.holdout_idx

    """ Read Berkeley Deep Drive label json. """
    def _load_data(self, drop_cls, force_num = None, hold_out_frac = None):
        # load annotation file
        data = pd.read_json(self.bdd_json_path)
        data = data.reset_index(drop = True)
        print(f">> Loading BDD data from {self.bdd_json_path}...")
        # complete path, since only images names in label file
        sep = "" if self.bdd_image_path[-1] == "/" else "/"
        data["name"] = self.bdd_image_path + sep + data["name"].astype(str)
        # drop other attributes
        data["weather"] = data.attributes.apply(lambda x: x["weather"])
        # drop other columns
        data = data.drop(columns = ["labels", "timestamp", "attributes"])
        # drop samples with unwanted classes
        data = data.loc[~data.weather.isin(drop_cls)]
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
        # debug: hold-out fraction
        data_idx = data.index.tolist()
        holdout_idx = []
        if hold_out_frac is not None:
            data_unique = data.drop_duplicates(subset = "name", keep = "first", inplace = False).sample(frac = 1, random_state = 123)
            data_unique, data_unique_holdout = train_test_split(data_unique, test_size = hold_out_frac)
            data_idx = data.loc[data.name.isin(data_unique.name.tolist())].index.tolist()
            holdout_idx = data.loc[data.name.isin(data_unique_holdout.name.tolist())].index.tolist()
            assert len(list(set(data_idx) & set(holdout_idx))) == 0
            assert len(list(set(data_idx) & set(data.index.tolist()))) + len(list(set(holdout_idx) & set(data.index.tolist()))) == len(data)
        return data, data_idx, holdout_idx
