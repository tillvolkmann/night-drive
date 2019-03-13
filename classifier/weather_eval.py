import os
import time
import torch
import glob
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import datasets.bdd.BDDWeatherDataset as bdd
import sklearn.metrics as metrics

# setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_f1_score(net, data_loader, num_batches = None):
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
        f1_score = metrics.f1_score(accumulated_targets, accumulated_outputs, average="weighted")
    net.train()
    return f1_score


if __name__ == "__main__":

    # set data paths
    path_valid_images = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid"
    path_valid_json = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid.json"
    dir_weights = "/home/SharedFolder/git/tillvolkmann/night-drive/20190313_ResNet18_train_A_over_448x448-32-traindevloss/*.pth"

    paths_weights = glob.glob(dir_weights)
    sort_idx = np.argsort([int(path_weights.split("epoch_")[-1].split(".pth")[0]) for path_weights in paths_weights])
    paths_weights = list(np.array(paths_weights)[sort_idx])

    # seeds
    torch.manual_seed(123)
    np.random.seed(123)

    # data transforms
    t_target_size = (448, 448)
    t_norm_mean = [0.485, 0.456, 0.406]
    t_norm_std = [0.229, 0.224, 0.225]
    transform = {
        "valid": transforms.Compose([
            transforms.Resize(t_target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=t_norm_mean, std=t_norm_std)])  # ImageNet
    }

    # create data sets
    ds_valid = bdd.BDDWeatherDataset(path_valid_json, path_valid_images, transform = transform["valid"])
    #ds_valid = bdd.BDDWeatherDataset(path_valid_json, path_valid_images, drop_cls = ["clear", "cloudy", "rainy"], transform = transform["valid"])
    #ds_valid = bdd.BDDWeatherDataset(path_valid_json, path_valid_images, transform=transform["valid"])
    #ds_valid.class_dict["clear"] = 0
    #ds_valid.class_dict["cloudy"] = 1
    #ds_valid.class_dict["rainy"] = 2
    #ds_valid.class_dict["snowy"] = 3

    # data loader
    dl_batch_size = 32
    dl_num_workers = 8
    dl_shuffle = True
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size = dl_batch_size, shuffle = dl_shuffle, num_workers = dl_num_workers)

    # create model
    #net = models.resnet50(pretrained = True)
    net = models.resnet18(pretrained = True)
    net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    net.fc = nn.Linear(net.fc.in_features, 4)

    # send model to device
    net.to(device)

    for path_weights in paths_weights:

        tic = time.time()

        # load weights
        net.load_state_dict(torch.load(path_weights)["model_state_dict"])

        # f1-score
        f1_valid = evaluate_f1_score(net, dl_valid)
        print(f"Done after {time.time() - tic:.2f}s -> F1-Score for data {path_valid_json.split(os.sep)[-1]} and weights {path_weights.split(os.sep)[-1]}: {f1_valid:.2f}")
