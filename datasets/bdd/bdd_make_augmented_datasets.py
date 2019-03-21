import os
from shutil import copyfile
import numpy as np
import pandas as pd
from PIL import Image
import BDDDataSets as bdd
from bdd_make_datasets import pandas_to_bddjson
import re


if __name__ == "__main__":

    # ==================================================================================================================
    # SETTINGS
    # ==================================================================================================================
    # root_dir_gan (set in config):
    #    DSR:             "/home/SharedFolder/CurrentDatasets"
    #    Till home:       "/home/till/SharedFolder/CurrentDatasets"
    # project_root (set in config):
    #    DSR: "/home/night-drive/Docker/SharedFolder/git/tillvolkmann/night-drive"
    #    Till home:       "/home/till/projects/night-drive"
    project_root = "/home/night-drive/Docker/SharedFolder/git/tillvolkmann/night-drive"

    # load config
    cfg_name = os.path.join(project_root, 'config/config_bdd_make_datasets.json')
    cfg = bdd.GetConfig(cfg_name)

    # auto paths, usually not necessary to change
    path_orig_images = os.path.join(cfg.root_dir_gan, "bdd100k/images/100k")
    path_main_json = os.path.join(cfg.root_dir_gan, "bdd100k_sorted/annotations/bdd100k_sorted_main.json")

    # ==================================================================================================================
    # load main json
    # ==================================================================================================================
    data = pd.read_json(path_main_json)
    data.reset_index(drop=True, inplace=True)

    # ==================================================================================================================
    # create a useful gan_info_dict containing info about each split element of sampler dict
    # ==================================================================================================================
    gan_info_dict = {}
    gan_info_dict["splits"] = cfg.gan_augment_dict.keys()
    c = 0
    for split in gan_info_dict["splits"]:
        gan_info_dict[split] = {}
        gan_info_dict[split]["split_traindev"] = re.sub("train", "train_dev", split)
        # get base split
        gan_info_dict[split]["base_split"] = cfg.gan_augment_dict[split]["base_split"]
        # get augmentation fraction
        gan_info_dict[split]["augment_fraction"] = cfg.gan_augment_dict[split]["augment_fraction"]
        # get augmentation mode
        gan_info_dict[split]["augment_mode"] = cfg.gan_augment_dict[split]["augment_mode"]
        # set association
        gan_info_dict[split]["set"] = "set_" + split[6:]
        #
        gan_info_dict[split]["aug_set_base"] = gan_info_dict[split]["set"]+"_base"
        gan_info_dict[split]["aug_set_n_over_aug"] = gan_info_dict[split]["set"]+"_n_over_aug"
        gan_info_dict[split]["aug_set_n_over_base"] = gan_info_dict[split]["set"]+"_n_over_base"
        #
        gan_info_dict[split]["base_set"] = "set_" + gan_info_dict[split]["base_split"][-1]
        gan_info_dict[split]["base_set_n_over"] = gan_info_dict[split]["base_set"]+"_n_over"
        # destination path
        if cfg.do_make_dirs_gan:  # create a separate dir for each split
            gan_info_dict[split]["destination_path"] = os.path.join(cfg.destination_path, split)
            gan_info_dict[split]["destination_path_traindev"] = os.path.join(cfg.destination_path, gan_info_dict[split]["split_traindev"])
        else:  # create all files in the same dir
            gan_info_dict[split]["destination_path"] = cfg.destination_path
        # destination file names
        gan_info_dict[split]["destination_json_filename"] = cfg.destination_filename_stem + split + ".json"
        gan_info_dict[split]["destination_json_over_filename"] = cfg.destination_filename_stem + split + "_over" + ".json"
        # for train_dev
        gan_info_dict[split]["destination_json_filename_traindev"] = cfg.destination_filename_stem + gan_info_dict[split]["split_traindev"] + ".json"
        gan_info_dict[split]["destination_json_over_filename_traindev"] = cfg.destination_filename_stem + gan_info_dict[split]["split_traindev"] + "_over" + ".json"
        # destination file path
        gan_info_dict[split]["destination_json_filepath"] = os.path.join(gan_info_dict[split]["destination_path"], gan_info_dict[split]["destination_json_filename"])
        gan_info_dict[split]["destination_json_over_filepath"] = os.path.join(gan_info_dict[split]["destination_path"], gan_info_dict[split]["destination_json_over_filename"])
        # for train_dev
        gan_info_dict[split]["destination_json_filepath_traindev"] = os.path.join(gan_info_dict[split]["destination_path"], gan_info_dict[split]["destination_json_filename_traindev"])
        gan_info_dict[split]["destination_json_over_filepath_traindev"] = os.path.join(gan_info_dict[split]["destination_path"], gan_info_dict[split]["destination_json_over_filename_traindev"])

        c += 1


    # ==================================================================================================================
    # Sample the data sets
    # ==================================================================================================================
    # set seed for numpy
    np.random.seed(1234)
    for split in gan_info_dict["splits"]:
        # unpack some helpers
        aug_set = gan_info_dict[split]["set"]
        aug_set_base = gan_info_dict[split]["aug_set_base"]
        aug_set_n_over_aug = gan_info_dict[split]["aug_set_n_over_aug"]
        aug_set_n_over_base = gan_info_dict[split]["aug_set_n_over_base"]
        base_split = gan_info_dict[split]["base_split"]
        base_set = gan_info_dict[split]["base_set"]
        base_set_n_over = gan_info_dict[split]["base_set_n_over"]
        # add columns to the dataframe indicating split association
        data[aug_set_base] = data[base_set]
        data[aug_set] = -1
        data[aug_set_n_over_aug] = 0
        data[aug_set_n_over_base] = 0
        #
        for tod, aug_frac in gan_info_dict[split]["augment_fraction"].items():  # for each timeofday
            for wc in sorted(data["weather"].unique().tolist()):  # for each weather condition
                for sub in ["train", "train_dev"]:
                    # get indices of all elements of the base set and given class
                    class_idx = data[(data[base_set].eq(sub) & data.timeofday.eq(tod) & data.weather.eq(wc))].index
                    # get number of original samples and number of oversamples
                    n_samples_total = len(class_idx)
                    n_over_total = data.loc[class_idx, base_set_n_over].sum()
                    #
                    if gan_info_dict[split]["augment_mode"] == "before_over":
                        # set identifier to 0 for included originals
                        data.loc[class_idx, aug_set] = 0
                        # get number of unique samples
                        n_samples_aug = np.round(n_samples_total * aug_frac).astype("int")
                        if n_samples_aug > 0:
                            # randomly sample from those indices without replacement for augmentation
                            idx_aug = np.random.choice(class_idx, size=n_samples_aug, replace=False)
                            data.loc[idx_aug, aug_set] = 1

                            # get number of oversamples for remaining base data and augmented data
                            n_over_aug = np.round(n_over_total * aug_frac).astype("int")
                            n_over_base = n_over_total - n_over_aug

                            # randomly over-sample the augmented data
                            idx_over_aug = np.random.choice(idx_aug, n_over_aug, replace=True)
                            idx_uni, counts = np.unique(idx_over_aug, return_counts=True)
                            data.loc[idx_uni, aug_set_n_over_aug] = counts

                            # randomly over-sample the base data
                            idx_base = np.setdiff1d(class_idx, idx_over_aug, assume_unique=True)
                            idx_over_base = np.random.choice(idx_base, n_over_base, replace=True)
                            idx_uni, counts = np.unique(idx_over_base, return_counts=True)
                            data.loc[idx_uni, aug_set_n_over_base] = counts

                    elif [gan_info_dict[split]["augment_mode"] == x for x in ["after_over", "after_over_expand"]]:
                        # get number of unique samples
                        n_samples_aug = np.round((n_samples_total + n_over_total) * aug_frac).astype("int")
                        if n_samples_aug > 0:

                            # randomly sample from those indices with replacement for augmentation
                            idx_aug = np.random.choice(class_idx, size=n_samples_aug, replace=True)
                            idx_uni_aug, counts = np.unique(idx_aug, return_counts=True)
                            data.loc[idx_uni_aug, aug_set] = 1
                            data.loc[idx_uni_aug, aug_set_n_over_aug] = np.maximum(0, counts - 1)

                            if gan_info_dict[split]["augment_mode"] == "after_over_expand":
                                # set identifier to 0 for included originals, which here is all
                                data.loc[class_idx, aug_set] = np.maximum(0, data.loc[class_idx, aug_set])
                                # here, every sample that is augmented is automatically also kept as original
                                data.loc[idx_aug_uni, aug_set] = 2
                            elif gan_info_dict[split]["augment_mode"] == "after_over":
                                #
                                n_samples_base = (n_samples_total + n_over_total) - n_samples_aug
                                # here, we keep first the max num of unqiues
                                n_unique_base = np.minimum(n_samples_base, n_samples_total)
                                idx_base = np.random.choice(class_idx, size=n_unique, replace=False)
                                data.loc[idx_base, aug_set] = data.loc[
                                                                  idx_base, aug_set] + 1  # currently, is either -1 or 1, so after this is 0 or 2
                                # randomly over-sample the base data
                                n_over_base = n_samples_base - n_unique_base
                                idx_over_base = np.random.choice(idx_base, n_over_base, replace=True)
                                idx_uni_base, counts = np.unique(idx_over_base, return_counts=True)
                                data.loc[idx_uni_base, aug_set_n_over_base] = counts


    # ==================================================================================================================
    # add suffix
    # ==================================================================================================================
    _, ext = os.path.splitext(data.loc[0,"name"])
    data["name_aug"] = ''
    mask = data["timeofday"]=="night"
    data.loc[mask, "name_aug"] = data.loc[mask, "name"].apply(lambda x: os.path.splitext(x)[0]+cfg.gan_transform_suffix["daytime"]+ext)
    mask = data["timeofday"]=="daytime"
    data.loc[mask, "name_aug"] = data.loc[mask, "name"].apply(lambda x: os.path.splitext(x)[0]+cfg.gan_transform_suffix["night"]+ext)

    # ==================================================================================================================
    # get paths
    # ==================================================================================================================
    data = data.reset_index(drop=True)
    # get paths to original BDD images and get paths to gan-augmented versions of the BDD images
    data["path"] = data["name"].apply(lambda x: os.path.join(path_orig_images, "train", x))
    data["path_aug"] = data["name_aug"].apply(
        lambda x: os.path.join(path_orig_images, "train" + cfg.gan_folder_suffix, x))
    for i in range(data.shape[0]):
        if not os.path.exists(data.loc[i, "path"]):
            data.loc[i, "path"] = os.path.join(path_orig_images, "val", data.loc[i, "name"])
            data.loc[i, "path_aug"] = os.path.join(path_orig_images, "val" + cfg.gan_folder_suffix,
                                                   data.loc[i, "name_aug"])
        if not i % 5000:
            print("Found path for {}-th image...".format(i))

    # ==================================================================================================================
    # now separate data, copy images, and create the different jsons
    # ==================================================================================================================
    _, ext = os.path.splitext(data.loc[0, "name"])
    for split in gan_info_dict["splits"]:  # for each split
        # Print message
        print("=== Processing data for split: {} ==================".format(split))
        # unpack some helpers
        aug_set = gan_info_dict[split]["set"]
        aug_set_base = gan_info_dict[split]["aug_set_base"]
        aug_set_n_over_aug = gan_info_dict[split]["aug_set_n_over_aug"]
        aug_set_n_over_base = gan_info_dict[split]["aug_set_n_over_base"]
        base_split = gan_info_dict[split]["base_split"]
        base_set = gan_info_dict[split]["base_set"]
        base_set_n_over = gan_info_dict[split]["base_set_n_over"]
        # for each sub-split
        for sub in ["train", "train_dev"]:
            if sub == "train_dev":
                destination_json_over_filepath = gan_info_dict[split]["destination_json_over_filepath_traindev"]
                destination_json_filepath = gan_info_dict[split]["destination_json_filepath_traindev"]
            else:
                destination_json_over_filepath = gan_info_dict[split]["destination_json_over_filepath"]
                destination_json_filepath = gan_info_dict[split]["destination_json_filepath"]

            ### get all elements associated with the current split into a separate data frame
            # first query all entries not to be augmented
            cur_file = data.query("({0}==@sub) & (({1}==0) | ({1}==2))".format(aug_set_base, aug_set)).reset_index(
                drop=True)
            # next, query the data to be augmented
            cur_aug = data.query("({0}==@sub) & (({1}==1) | ({1}==2))".format(aug_set_base, aug_set)).reset_index(
                drop=True)
            # rename augmented data
            cur_aug["name"] = cur_aug["name_aug"]
            cur_aug["path"] = cur_aug["path_aug"]
            cur_aug[aug_set_n_over_base] = cur_aug[aug_set_n_over_aug]
            # now combine
            cur_file = pd.concat([cur_file, cur_aug], axis=0).reset_index(drop=True)

            # create folder structure
            if not os.path.exists(gan_info_dict[split]["destination_path"]):
                os.makedirs(gan_info_dict[split]["destination_path"])
            elif cfg.do_make_dirs_gan:
                raise Exception("Destination folder(s) already exist.")

            # save a json in bdd format containing only the unique original and augmented images
            if cfg.do_make_jsons_gan:
                print("Writing json to {}".format(destination_json_filepath))
                pandas_to_bddjson(cur_file.copy(), destination_json_filepath)

            # copy original images associated with current split into new folder
            if cfg.do_copy_images_gan == True:
                print("Copying {} images to {}".format(cur_file.shape[0], gan_info_dict[split]["destination_path"]))
                for img in cur_file["path"]:
                    img_path = os.path.join(img)
                    copyfile(img_path,
                             os.path.join(gan_info_dict[split]["destination_path"], os.path.basename(img_path)))

            # append oversamples to cur_file (file names = original file name + copy1, copy2, etc.
            col_name = aug_set_n_over_base
            if cfg.do_oversample_physically:
                cf_shape_before = cur_file.shape[0]
                for i in range(cf_shape_before):
                    n_over = int(cur_file.loc[i, col_name])
                    for j in range(n_over):
                        cur_file.loc[cur_file.shape[0], :] = cur_file.loc[i, :]
                        cur_file = cur_file.reset_index(drop=True)
                        # pd.concat([cur_file,cur_file.iloc[i,:]], ignore_index=True)
                        # make physical copies of the oversamples, if requested
                        if cfg.do_oversample_physically == True:
                            name_original = cur_file.loc[i, "name"]
                            name_copy = os.path.join(gan_info_dict[split]["destination_path"], os.path.basename(
                                name_original.split(".")[0] + "_copy" + str(j + 1) + "." + name_original.split(".")[
                                    1]))  # rename file by appending _copy1, _copy2, etc
                            print(name_original, '>>>\n', name_copy)
                            copyfile(name_original, name_copy)
                            cur_file.loc[cur_file.shape[0] - 1, "name"] = name_copy  # store the new name
                    if i % 1000 == 0:
                        print("Over-sampling done for {} of {} entries.".format(i, cf_shape_before))
            else:  # this is much faster
                cur_file.reset_index(drop=True, inplace=True)
                cur_file = pd.concat([cur_file,
                                      cur_file.loc[np.repeat(cur_file.index.values, cur_file[col_name])]],
                                     axis=0).reset_index(drop=True)

            # shuffle
            # cur_file.sample(frac=1.0, random_state=123).reset_index(drop=True)

            # save a json in bdd format containing also the over-samples
            if cfg.do_make_jsons_gan:
                pandas_to_bddjson(cur_file.copy(), destination_json_over_filepath)

    # ==================================================================================================================
    # WRITE MAIN JSON
    # ==================================================================================================================
    # write a main json containing the combined information for all splits, including additional info columns
    # first get relative path for images
    # data.name = data.name.map(os.path.basename)
    # now write
    if cfg.version == 0:
        data.to_json(os.path.join(cfg.destination_path, cfg.destination_filename_stem + "main" + ".json"))
    elif cfg.version == 1:
        data.to_json(os.path.join(cfg.destination_path, cfg.destination_filename_stem + "main" + ".json"),
                     orient="records")
