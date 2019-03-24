import torch
import torch.nn as nn
import torchvision.models as models
from datasets.bdd.BDDWeatherDataset import *
import albumentations.augmentations.transforms as transforms
from albumentations import IAAAdditiveGaussianNoise
from torch.utils.data.sampler import SubsetRandomSampler
from albumentations import Compose
from albumentations.pytorch import ToTensor


def weather_classifier(path_train_json = None, path_train_images = None, path_valid_json = None, path_valid_images = None):

    # seeds
    torch.manual_seed(123)
    np.random.seed(123)

    # data loader
    dl_batch_size = 128
    dl_num_workers_train = 10
    dl_num_workers_valid = 6
    dl_shuffle = True

    # data transforms
    t_original_size = (720, 1280)
    t_target_size = (224, 224)
    t_norm_mean = [0.485, 0.456, 0.406]  # ImageNet
    t_norm_std = [0.229, 0.224, 0.225]  # ImageNet
    transform = {
        "train": Compose([
            transforms.RandomBrightnessContrast(brightness_limit = 0.2, contrast_limit = 0.2, p = 0.5),
            IAAAdditiveGaussianNoise(p = 0.5),
            transforms.HorizontalFlip(p = 0.5),
            transforms.RandomSizedCrop(min_max_height = (t_original_size[0] // 2, t_original_size[0]),
                                       height = t_target_size[0], width = t_target_size[1], w2h_ratio = 1.777778),
            transforms.Resize(height = t_target_size[0], width = t_target_size[1]),
            ToTensor(normalize = {"mean": t_norm_mean, "std": t_norm_std})]),
        "valid": Compose([
            transforms.Resize(height = t_target_size[0], width = t_target_size[1]),
            ToTensor(normalize = {"mean": t_norm_mean, "std": t_norm_std})])
    }

    # create data sets and data loader
    dl_train = None
    dl_valid = None
    class_dict = None
    num_classes = None
    if path_train_json is not None and path_train_images is not None:
        ds_train = BDDWeatherDataset(path_train_json, path_train_images, transform = transform["train"], drop_cls = ["cloudy"])
        #ds_train = BDDWeatherDataset(path_train_json, path_train_images, transform = transform["train"])
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size = dl_batch_size, shuffle = dl_shuffle, num_workers = dl_num_workers_train)
        class_dict = ds_train._get_class_dict()
        num_classes = ds_train._get_num_classes()
    if path_valid_json is not None and path_valid_images is not None:
        ds_valid = BDDWeatherDataset(path_valid_json, path_valid_images, transform = transform["valid"], drop_cls = ["cloudy"])
        #ds_valid = BDDWeatherDataset(path_valid_json, path_valid_images, transform = transform["valid"])
        dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size = dl_batch_size, shuffle = dl_shuffle, num_workers = dl_num_workers_valid)
        if class_dict is not None and num_classes is not None:
            assert class_dict == ds_valid._get_class_dict()
            assert num_classes == ds_valid._get_num_classes()
        else:
            class_dict = ds_valid._get_class_dict()
            num_classes = ds_valid._get_num_classes()

    # create model
    net = models.resnet18(pretrained = True)
    # Adaptive Pooling needed for resolutions > 224 x 224
    net.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout(p = 0.1))
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    return net, dl_train, dl_valid, class_dict
