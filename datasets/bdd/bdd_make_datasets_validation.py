import os
import pandas as pd

gan_transform_suffix = {"daytime": "_transfer_BtoA", "night": "_transfer_AtoB"}
json_transform_suffix = "_converted"
root_path = "/home/till/SharedFolder/CurrentDatasets/bdd100k"
# dor every esplit
for split in ["valid", "train"]:
    # load original json
    if split == "train":
        path_original_json = os.path.join(root_path, "labels/bdd100k_labels_images_train.json")
    elif split == "valid":
        path_original_json = os.path.join(root_path, "labels/bdd100k_labels_images_val.json")
    data = pd.read_json(path_original_json)
    data.reset_index(drop=True, inplace=True)
    # convert_filenames
    _, ext = os.path.splitext(data.loc[0,"name"])
    mask = data.attributes.apply(lambda x: x["timeofday"]=="night")
    data.loc[mask, "name"] = data.loc[mask, "name"].apply(lambda x: os.path.splitext(x)[0]+gan_transform_suffix["daytime"]+ext)
    mask = data.attributes.apply(lambda x: x["timeofday"]=="daytime")
    data.loc[mask, "name"] = data.loc[mask, "name"].apply(lambda x: os.path.splitext(x)[0]+gan_transform_suffix["night"]+ext)
    # save converterd json (in original directory)
    dir_original_json, name_original_json = os.path.split(path_original_json)
    name_original_json = os.path.splitext(name_original_json)[0]
    name_converted_json = name_original_json+json_transform_suffix+".json"
    path_converted_json = os.path.join(dir_original_json, name_converted_json)
    data.to_json(path_or_buf=path_converted_json, orient="records")
