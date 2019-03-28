import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BDDTimeOfDayDataset(Dataset):

    """ Constructor """
    def __init__(self, root_dir, transform=None, augment=None, split="bddtrain", dropcls=None,
                 force_num=None, with_labels=True, output_filenames=False, which_ganmodel="v032_e14"):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.with_labels = with_labels
        self.output_filenames = output_filenames
        self.which_ganmodel = which_ganmodel
        self.data = self._load_data(split, dropcls, force_num)
        if with_labels:
            self.class_dict = dict(zip(list(np.sort(self.data.timeofday.unique())), list(range(self._get_num_classes()))))
        else:
            self.class_dict = {}

    """ Provides the size of the dataset """
    def __len__(self):
        return self.data.shape[0]

    """ Provides one sample (image + target) from index """
    def __getitem__(self, ix):
        # load image and target
        img = self._load_image(ix)
        # transform images
        if self.transform is not None:
            img = self.transform(img)
        # augment image
        if self.augment is not None:
            img = self.augment(img)
        if self.with_labels:
            target = self.class_dict[self._get_target(ix)]
            if self.output_filenames:
                return img, target, self.data.name[ix]
            else:
                return img, target
        else:
            return img, self.data.name[ix]

    """ Load one image from disk as PIL image """
    def _load_image(self, ix):
        img_path = self.data.name[ix]
        img = Image.open(img_path)
        return img

    """ Get one target (timeofday classification) """
    def _get_target(self, ix):
        target = self.data.iloc[ix]["timeofday"]
        return target

    """ Get target value count """
    def _get_class_counts(self):
        return self.data.timeofday.value_counts().to_dict()

    """ Get number of classes """
    def _get_num_classes(self):
        return len(self.data.timeofday.unique())

    """ Get class-int mapping """
    def _get_class_dict(self):
        return self.class_dict

    def _load_data(self, split, dropcls, force_num):
        """
        Read Berkeley Deep Drive label json.

        Requires original BDD folder structure as well as equally-named json files.

        """
        if split == "bddtrain":
            print(">> Using original BDD training set")
            if self.with_labels:
                # training set
                data = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_train.json"))
                # concat path, since different for training and val set
                data["name"] = self.root_dir + "/images/100k/train/" + data["name"].astype(str)
            else:
                files = glob.glob(self.root_dir + "/images/100k/train/" + "/**/*.jpg", recursive = True) \
                        + glob.glob(self.root_dir + "/images/100k/train/" + "/**/*.png", recursive = True)
                data = pd.DataFrame(files, columns=["name"])
        elif split == "bddvalid":
            print(">> Using original BDD validation set")
            if self.with_labels:
                # validation set
                data = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_val.json"))
                # concat path, since different for training and val set
                data["name"] = self.root_dir + "/images/100k/val/" + data["name"].astype(str)
            else:
                files = glob.glob(self.root_dir + "/images/100k/val/" + "/**/*.jpg", recursive = True) \
                        + glob.glob(self.root_dir + "/images/100k/val/" + "/**/*.png", recursive = True)
                data = pd.DataFrame(files, columns=["name"])
        elif split == "bddtest":
            print(">> Using original BDD test set")
            if self.with_labels:
                # test set
                data = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_test.json"))
                # concat path, since different for training and val set
                data["name"] = self.root_dir + "/images/100k/test/" + data["name"].astype(str)
            else:
                files = glob.glob(self.root_dir + "/images/100k/test/" + "/**/*.jpg", recursive = True) \
                        + glob.glob(self.root_dir + "/images/100k/test/" + "/**/*.png", recursive = True)
                data = pd.DataFrame(files, columns=["name"])
        elif split == "bddtrainvalid":
            print(">> Mixing original BDD training and validation sets")
            if self.with_labels:
                # training set
                data_train = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_train.json"))
                # concat path, since different for training and val set
                data_train["name"] = self.root_dir + "/images/100k/train/" + data_train["name"].astype(str)
                # validation set
                data_val = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_val.json"))
                # concat path, since different for training and val set
                data_val["name"] = self.root_dir + "/images/100k/val/" + data_val["name"].astype(str)
                data = pd.concat([data_train, data_val], axis = 0)
            else:
                files = glob.glob(self.root_dir + "/images/100k/train/" + "/**/*.jpg", recursive = True) \
                        + glob.glob(self.root_dir + "/images/100k/train/" + "/**/*.png", recursive = True) \
                        + glob.glob(self.root_dir + "/images/100k/val/" + "/**/*.jpg", recursive = True) \
                        + glob.glob(self.root_dir + "/images/100k/val/" + "/**/*.png", recursive = True)
                data = pd.DataFrame(files, columns=["name"])
        elif split == "bddtest_converted":
            print(">> Using GAN-transformed BDD test set")
            if self.with_labels:
                # test set
                data = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_test_converted.json"))
                # concat path, since different for training and val set
                data["name"] = self.root_dir + "/images/100k/test_converted_" + self.which_ganmodel + "/" + data["name"].astype(str)
            else:
                files = glob.glob(self.root_dir + "/images/100k/test_converted_" + self.which_ganmodel + "/**/*.jpg", recursive=True) \
                        + glob.glob(self.root_dir + "/images/100k/test_converted_" + self.which_ganmodel + "/**/*.png", recursive=True)
                data = pd.DataFrame(files, columns=["name"])
        elif split == "bddvalid_converted":
            print(">> Using GAN-transformed BDD validation set")
            if self.with_labels:
                # validation set
                data = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_val_converted.json"))
                # concat path, since different for training and val set
                data["name"] = self.root_dir + "/images/100k/val_converted_" + self.which_ganmodel + "/" + data["name"].astype(str)
            else:
                files = glob.glob(self.root_dir + "/images/100k/val_converted_" + self.which_ganmodel + "/**/*.jpg", recursive=True) \
                        + glob.glob(self.root_dir + "/images/100k/val_converted_" + self.which_ganmodel + "/**/*.png", recursive=True)
                data = pd.DataFrame(files, columns=["name"])
        elif split == "bddtrain_converted":
            print(">> Using GAN-transformed BDD train set")
            if self.with_labels:
                # training set
                data = pd.read_json(os.path.join(self.root_dir, "labels/bdd100k_labels_images_train_converted.json"))
                # concat path, since different for training and val set
                data["name"] = self.root_dir + "/images/100k/train_converterd_" + self.which_ganmodel + "/" + data["name"].astype(str)
            else:
                files = glob.glob(self.root_dir + "/images/100k/train_converterd_" + self.which_ganmodel + "/**/*.jpg", recursive=True) \
                        + glob.glob(self.root_dir + "/images/100k/train_converterd_" + self.which_ganmodel + "/**/*.png", recursive=True)
                data = pd.DataFrame(files, columns=["name"])
        else:
            raise Exception(f"Split {split} is unknown.")

        if self.with_labels:
            # drop other attributes
            data["timeofday"] = data.attributes.apply(lambda x: x["timeofday"])
            # drop other columns
            data = data.drop(columns = ["labels", "timestamp", "attributes"])
            # drop samples with unwanted classes
            if dropcls is None:
                dropcls = []
            data = data.loc[~data.timeofday.isin(dropcls)]
            data = data.sample(frac = 1, random_state=123).reset_index(drop = True)
            # balance classes to fixed number
            if force_num is not None:
                data_balanced = pd.DataFrame()
                for timeofday, count in data.loc[:, "timeofday"].value_counts().to_dict().items():
                    if force_num >= count:
                        orig = data.loc[data.timeofday == timeofday]
                        # sample with replacement
                        over = orig.iloc[np.random.randint(0, orig.shape[0], size = force_num - orig.shape[0])]
                        data_balanced = pd.concat([data_balanced, orig, over], axis = 0)
                    else:
                        orig = data.loc[data.timeofday == timeofday].iloc[0:force_num]
                        data_balanced = pd.concat([data_balanced, orig], axis = 0)
                data = data_balanced.reset_index(drop = True)
        return data
