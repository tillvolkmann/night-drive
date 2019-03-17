import os
import glob
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import sklearn.metrics as metrics
import datasets.bdd.BDDWeatherDataset as bdd
import albumentations.augmentations.transforms as transforms
from albumentations import Compose
from albumentations.pytorch import ToTensor

# setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(net, data_loader, score_type = "f1_score", num_batches = None):
    # f1 score: https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel
    net.eval() # disables dropout, etc.
    with torch.no_grad(): # temporarily disables gradient computation for speed-up
        accumulated_targets = []
        accumulated_outputs = []
        for i, data in enumerate(data_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            accumulated_targets.extend(targets.detach().cpu().numpy().tolist())
            accumulated_outputs.extend(np.argmax(outputs.detach().cpu().numpy().tolist(), axis = 1))
            if num_batches is not None and (i >= (num_batches - 1)):
                break
        if score_type == "f1_score":
            score = metrics.f1_score(accumulated_targets, accumulated_outputs, average="weighted")
        elif score_type == "accuracy":
            score = metrics.accuracy_score(accumulated_targets, accumulated_outputs)
        else:
            raise Exception("Unknown score_type")
    net.train()
    return score


if __name__ == "__main__":

    # set data paths
    path_valid_images = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_dev_A"
    path_valid_json = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_dev_A_over.json"
    dir_weights = "./models/resnet18_weather_classifier_train_A_over*.pth"

    paths_weights = glob.glob(dir_weights)
    sort_idx = np.argsort([int(path_weights.split("epoch_")[-1].split(".pth")[0]) for path_weights in paths_weights])
    paths_weights = list(np.array(paths_weights)[sort_idx])

    # seeds
    torch.manual_seed(123)
    np.random.seed(123)

    # data transforms
    t_original_size = (720, 1280)
    t_target_size = (448, 448)
    t_norm_mean = [0.485, 0.456, 0.406] # ImageNet
    t_norm_std = [0.229, 0.224, 0.225] # ImageNet
    transform = {
        "valid": Compose([
            #transforms.RandomSizedCrop(min_max_height = (int(t_original_size[0] * 0.75), t_original_size[0]), height = t_target_size[0], width = t_target_size[1], w2h_ratio = 1.777778),
            transforms.Resize(height = t_target_size[0], width = t_target_size[1]),
            ToTensor(normalize = {"mean": t_norm_mean, "std": t_norm_std})])
    }

    # create data sets
    ds_valid = bdd.BDDWeatherDataset(path_valid_json, path_valid_images, transform = transform["valid"], drop_cls = ["cloudy"])

    # data loader
    dl_batch_size = 32
    dl_num_workers_valid = 8
    dl_shuffle = True
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size = dl_batch_size, shuffle = dl_shuffle, num_workers = dl_num_workers_valid)

    # create model
    net = models.resnet18(pretrained = True)
    net.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Needed for resolutions > 224 x 224
    net.fc = nn.Linear(net.fc.in_features, ds_valid._get_num_classes())

    # send model to device
    net.to(device)

    for path_weights in paths_weights:

        tic = time.time()

        # load weights
        net.load_state_dict(torch.load(path_weights)["model_state_dict"])

        # f1-score
        f1_valid = evaluate(net, dl_valid)
        print(f"Done after {time.time() - tic:.2f}s -> F1-Score for data {path_valid_json.split(os.sep)[-1]} and weights {path_weights.split(os.sep)[-1]}: {f1_valid:.2f}")
