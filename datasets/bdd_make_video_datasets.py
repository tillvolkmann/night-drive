import sys
from bdd_make_datasets import pandas_to_bddjson
import pandas as pd
import os

# Set root dir of BDD 100k data set
which_host = 'dsr' # Options ['till', 'dsr']
if which_host == 'till':
    bdd_100k_root = "/home/till/SharedFolder/CurrentDatasets/bdd100k/"
    project_root = "/home/till/projects/night-drive/"
elif which_host == 'dsr':
    bdd_100k_root = "/home/SharedFolder/CurrentDatasets/bdd100k/"
    project_root = "/home/SharedFolder/git/tillvolkmann/night-drive/"

# get some more paths
video_root = os.path.join(bdd_100k_root, "videos")
json_path_train = os.path.join(bdd_100k_root, "labels/bdd100k_labels_images_train.json")
image_root_train = os.path.join(bdd_100k_root, "images/100k/train")
json_path_val = os.path.join(bdd_100k_root, "labels/bdd100k_labels_images_val.json")
image_root_val = os.path.join(bdd_100k_root, "images/100k/val")

# get the labels for all original frames
sys.path.append(project_root)
from utils import eval_utils
df = eval_utils.load_bdd_json(json_path_train, image_root_train)
df = pd.concat([df, eval_utils.load_bdd_json(json_path_val, image_root_val)], axis=0)
df.reset_index(drop=True, inplace=True)

# get the available videos' names
df_vid = pd.DataFrame(columns=df.columns)
# verify that all file names listed in json are somewhere within the root directory
for i, name in enumerate(df.name):
    name_vid = os.path.splitext(name)[0]+".mov"
    path = eval_utils.get_filepath(video_root, name_vid, notfounderror=False)
    if path is not None:
        df_vid = df_vid.append(df.iloc[i], ignore_index=True)
        df_vid.loc[df_vid.shape[0]-1, "name"] = name_vid

# save json containing only all available video files
json_path_vid = os.path.join(video_root, "bdd100k_labels_videos.json")
pandas_to_bddjson(df_vid, json_path_vid)