import os
import glob
import time
import torch
import numpy as np
from weather_classifier import weather_classifier
from classification_eval import evaluate_weather

# setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    # set data paths
    path_valid_images = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test"
    path_valid_json = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_night.json"
    dir_weights = "/home/SharedFolder/trained_models/night-drive/weather_classifier/train_C_over/*.pth"

    paths_weights = glob.glob(dir_weights)
    sort_idx = np.argsort([int(path_weights.split("epoch_")[-1].split(".pth")[0]) for path_weights in paths_weights])
    paths_weights = list(np.array(paths_weights)[sort_idx])

    # create classifer, datasets and dataloader
    net, _, dl_valid, _ = weather_classifier(path_valid_json = path_valid_json, path_valid_images = path_valid_images)

    # send model to device
    net.to(device)

    for path_weights in paths_weights:

        tic = time.time()

        # load weights
        net.load_state_dict(torch.load(path_weights)["model_state_dict"])

        # f1-score
        f1_valid, _ = evaluate_weather(net, dl_valid, score_type = "f1_score_weighted")
        print(f"Done after {time.time() - tic:.2f}s -> F1-Score for data {path_valid_json.split(os.sep)[-1]} and weights {path_weights.split(os.sep)[-1]}: {f1_valid:.5f}")
