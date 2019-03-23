import os
import glob
import time
import torch
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from weather_classifier import weather_classifier
from classification_eval import evaluate_weather

# setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir_weights", dest = "dir_weights", help = "dir with weights to be used for evaluation")
    parser.add_argument("--path_valid_json", dest = "path_valid_json", help = "path to json file of dataset to be used for evaluation")
    parser.add_argument("--path_valid_images", dest = "path_valid_images", help = "path to images of dataset to be used for evaluation")
    parser.add_argument("--out_folder", dest = "out_folder", help = "folder for output files")
    args = parser.parse_args()

    # set data paths
    path_valid_images = args.path_valid_images
    path_valid_json = args.path_valid_json
    dir_weights = args.dir_weights
    out_folder = args.out_folder

    out_file = dir_weights.split(os.sep)[-1] + "_ON_" + path_valid_json.split(os.sep)[-1].split(".json")[0] + ".csv"
    paths_weights = glob.glob(dir_weights + "/*.pth")
    sort_idx = np.argsort([int(path_weights.split("epoch_")[-1].split(".pth")[0]) for path_weights in paths_weights])
    paths_weights = list(np.array(paths_weights)[sort_idx])

    # create classifer, datasets and dataloader
    net, _, dl_valid, class_dict = weather_classifier(path_valid_json = path_valid_json, path_valid_images = path_valid_images)

    # send model to device
    net.to(device)

    df = pd.DataFrame()

    tic = time.time()

    for path_weights in paths_weights:

        # load weights
        net.load_state_dict(torch.load(path_weights)["model_state_dict"])

        # scores
        scores, values = evaluate_weather(net, dl_valid,
                                          score_types=["f1_score_micro", "f1_score_macro", "f1_score_weighted",
                                                       "accuracy", "roc_auc_micro", "roc_auc_macro", "pr_micro",
                                                       "pr_macro"], cut_off = None, class_dict = class_dict)

        # update
        df_ = pd.DataFrame.from_dict(scores, orient = "index").T
        df_.insert(loc = 0, column = "epoch", value = int(path_weights.split("epoch_")[-1].split(".pth")[0]))
        df_.insert(loc = 1, column = "weights", value = path_weights)
        df_.insert(loc = 2, column = "data", value = path_valid_json)
        df = pd.concat([df, df_], axis = 0)

    df = df.reset_index(drop = True)
    df.to_csv(out_folder + "/" + out_file, header = True, index = False)

    print(f"Done after {time.time() - tic:.2f}s.")
