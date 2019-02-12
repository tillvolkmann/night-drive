import os
import glob
import pandas as pd
from skimage import io

base_path = "/home/SharedFolder/CurrentDatasets/mapillary-vistas-dataset_public_v1.1"
dataset_paths = ["testing", "training", "validation"]

df = pd.DataFrame(columns = ["set", "file", "width", "height"])
for dataset_path in dataset_paths:
    image_paths = glob.glob(str(os.path.join(base_path, dataset_path)) + '/**/*.jpg', recursive = True)
    for image_path in image_paths:
        img = io.imread(image_path)
        df = df.append({"set": dataset_path, "file": image_path.split(os.sep)[-1], "width": img.shape[1], "height": img.shape[0]}, ignore_index = True)
df.to_csv("mapillary_aspectratios.csv", index = False)
