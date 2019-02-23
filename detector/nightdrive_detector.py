import time
import torch
import neptune
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from datasets.bdd.NightDriveDataset import DetectorDataset
import sklearn.metrics as metrics


def get_config(filename):
    """
    Loads config file for classifier and detector training.
    Available files:
        ~ 'config_bdd_setA.json' : 100% day, 0% night
        ~ 'config_bdd_setB.json' : 75% day, 25% night
        ~ 'config_bdd_setC.json' : 50% day, 50% night
    """
    import json
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


if __name__ == '__main__':

    # get config
    config_file = 'config_bdd_setA.json'
    cfg = get_config(config_file)  # see docstring for info on available config files

    # seeds
    torch.manual_seed(123)
    np.random.seed(123)

    # setup job monitoring on Neptune
    ctx = neptune.Context()

    # setup job monitoring on TensorBoardX
    #writer = SummaryWriter()

    # setting device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # show validation loss
    calc_valid_loss = True

    # data transforms
    t_target_size = (224, 224)
    t_norm_mean = [0.485, 0.456, 0.406]
    t_norm_std = [0.229, 0.224, 0.225]
    transform = {
        "train": transforms.Compose([
            transforms.Resize(t_target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=t_norm_mean, std=t_norm_std)]),  # ImageNet
        "valid": transforms.Compose([
            transforms.Resize(t_target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=t_norm_mean, std=t_norm_std)])  # ImageNet
    }

    # create data sets
    ds_train = DetectorDataset(cfg.root_dir, database=cfg.database, split="train", transform=transform["train"],
                                            sampler_dict=cfg.sampler_dict, dropclass_dict=cfg.dropclass_dict,
                                            mergeclass_dict=cfg.mergeclass_dict)
    ds_train_dev = DetectorDataset(cfg.root_dir, database=cfg.database, split="train_dev", transform=transform["train"],
                                                sampler_dict=cfg.sampler_dict, dropclass_dict=cfg.dropclass_dict,
                                                mergeclass_dict=cfg.mergeclass_dict)
    ds_valid = DetectorDataset(cfg.root_dir, database=cfg.database, split="valid", transform=transform["valid"],
                                            sampler_dict=cfg.sampler_dict, dropclass_dict=cfg.dropclass_dict,
                                            mergeclass_dict=cfg.mergeclass_dict)
    ds_test = DetectorDataset(cfg.root_dir, database=cfg.database, split="test", transform=transform["valid"],
                                           sampler_dict=cfg.sampler_dict, dropclass_dict=cfg.dropclass_dict,
                                           mergeclass_dict=cfg.mergeclass_dict)


    # data loader
    dl_batch_size = 28
    dl_num_workers = 8
    dl_shuffle = True
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=dl_batch_size, shuffle=dl_shuffle, num_workers=dl_num_workers)
    dl_train_dev = torch.utils.data.DataLoader(ds_train_dev, batch_size=dl_batch_size, shuffle=dl_shuffle, num_workers=dl_num_workers)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=dl_batch_size, shuffle=dl_shuffle, num_workers=dl_num_workers)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=dl_batch_size, shuffle=dl_shuffle, num_workers=dl_num_workers)
