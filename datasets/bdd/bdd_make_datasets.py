import os
from shutil import copyfile
import numpy as np
import pandas as pd
from PIL import Image
import BDDDataSets as bdd
import warnings


class InsufficientSamplesWarning(UserWarning):
    pass


def stratified_sampler(cross_total, cross_avail, sampler_dict, verbose=1):
    # initialize empty sampler table
    sampler_tab = cross_total.copy()
    sampler_tab[:] = 0

    # get some helpers
    _weather_classes = sampler_tab.columns
    _num_weather_classes = len(_weather_classes)
    _timeofday_classes = sampler_tab.index

    # First, we need to apply the class_min threshold
    if sampler_dict["class_min"] is not None:
        for tod in _timeofday_classes:
            if sampler_dict["class_dist"][tod] > 0:
                sampler_tab.loc[tod] = cross_avail.loc[tod].apply(
                    lambda x: np.minimum(x, sampler_dict["class_min"]))  # minimum of requested and available samples
                sampler_tab.loc[tod] = np.minimum(sampler_tab.loc[tod], cross_avail.loc[tod])
    # Second, we sample the remaining images according to the specified distribution across weather classes
    if sampler_dict["balancing"] == "none":
        # day and night eq to their respective dist
        _dist_weather_classes = cross_total.div(cross_total.sum(axis=1), axis=0)

    elif sampler_dict["balancing"] == "like-day-and-night":
        # day and night eq to their combined dist (note this is not weightes by number of samples among timeofday categories, i.e. the weather dists during day and night have equal weight)
        _dist_weather_classes = cross_total.copy()
        _dist_weather_classes.loc["daytime"] = cross_total.div(cross_total.sum(axis=1), axis=0).mean(axis=0)
        _dist_weather_classes.loc["night"] = _dist_weather_classes.loc["daytime"]

    elif sampler_dict["balancing"] == "like-day":
        # day and night eq to day's dist
        _dist_weather_classes = cross_total.copy()
        _dist_weather_classes.loc["daytime"] = cross_total.loc["daytime"].div(cross_total.loc["daytime"].sum(), axis=0)
        _dist_weather_classes.loc["night"] = _dist_weather_classes.loc["daytime"]

    elif sampler_dict["balancing"] in ["max-each", "max-adjusted", "max-adjusted-unequal"]:
        # maximum balance within each time of day subgroup
        _dist_weather_classes = cross_total.copy()
        _dist_weather_classes[:] = 1 / _num_weather_classes

    print(_dist_weather_classes)
    # Third, convert distributions to requested numbers of images
    #     "over": "before" ~ assumes oversampling before counting
    #     "over": "after" ~ assumes oversampling after counting
    if sampler_dict["over"] == "before":
        # assumes oversampling is done before sample collection; i.e., split set will have specified fraction of "resampled" images of each timeofday
        for tod in _timeofday_classes:
            # get target number of samples per weather class (after oversampling)
            _class_size = sampler_dict["n"] * sampler_dict["class_dist"][tod] / _num_weather_classes
            # get num of original images to draw to match class target size and specified distibution
            _num_originals = _class_size * _dist_weather_classes.loc[tod] / _dist_weather_classes.loc[
                tod].max()  # note the scaling of the weather dist to a [0 1] range, so that the max class exactly fills the target while preserving the desired distribution
            # assign to sampler table (accounting for already assigned data from thresholding)
            sampler_tab.loc[tod] = np.maximum(sampler_tab.loc[tod], _num_originals)

    elif sampler_dict["over"] in ["none", "after"]:
        # assumes oversampling is done after sample collection; i.e., split set will have specified fraction of "unqiue" images of each timeofday
        for tod in _timeofday_classes:
            print(_dist_weather_classes.loc[tod])
            _num_originals = sampler_dict["n"] * sampler_dict["class_dist"][tod] * _dist_weather_classes.loc[tod]
            # correct for class_min threshold
            _temp_diff = _num_originals - sampler_tab.loc[tod]
            _bias = _temp_diff.where(_temp_diff < 0).fillna(0).sum()
            _capped_bool = _temp_diff < 0
            _temp_dist = _dist_weather_classes.loc[tod]
            _temp_dist[_capped_bool] = 0
            _temp_dist = _temp_dist / _temp_dist.sum()
            _num_originals = _num_originals + _temp_dist * _bias
            # here, we could correct for availability if wanted, though this would alter the requested distribution
            # _do_correct = True
            # if _do_correct:
            # now assign
            sampler_tab.loc[tod] = np.maximum(sampler_tab.loc[tod], _num_originals)

    # max balancing but adjust so that proportions among sets of different sizes (e.g. 25% night and 50%) are preserved
    # "max-adjusted" preserves with respect to 100% sample size, while "max-adjusted-unequal" preserves with respect to the max percentage sample size in the experiment (i.e. 50% night)
    if sampler_dict["balancing"] in ["max-adjusted"]:
        sampler_tab.loc["daytime"] = np.minimum(sampler_tab.loc["daytime"],
                                                cross_avail.loc["daytime"] * sampler_dict["class_dist"]["daytime"])
        sampler_tab.loc["night"] = np.minimum(sampler_tab.loc["night"],
                                              cross_avail.loc["night"] * sampler_dict["class_dist"]["night"])
    if sampler_dict["balancing"] in ["max-adjusted-unequal"]:
        sampler_tab.loc["daytime"] = np.minimum(sampler_tab.loc["daytime"],
                                                cross_avail.loc["daytime"] * sampler_dict["class_dist"]["daytime"])
        sampler_tab.loc["night"] = np.minimum(sampler_tab.loc["night"],
                                              cross_avail.loc["night"] * sampler_dict["class_dist"]["night"] * 1 / 0.5)

    # correct for actually available numbers
    if (sampler_tab > cross_avail).any().any():
        # issue a warning
        warnings.warn('Number of available sample images is smaller than number of requested images.',
                      InsufficientSamplesWarning)
        print("\n====================================================")
        print("States during InsufficientSamplesWarning from stratified_sampler")
        print("====================================================")
        print("\nsampler_tab during Warning")
        print(sampler_tab)
        print("\ncross_avail during Warning")
        print(cross_avail)
        print("\nsampler_tab > cross_avail during Warning")
        print(sampler_tab > cross_avail)
        # make correction
        sampler_tab[sampler_tab > cross_avail] = cross_avail[sampler_tab > cross_avail]

        # Fourth, determine n to over-sample
    over_tab = sampler_tab.copy()
    over_tab[:] = 0
    if sampler_dict["over"] != "none":
        over_tab.loc["daytime"] = sampler_tab.loc["daytime"].max() - sampler_tab.loc["daytime"]
        over_tab.loc["night"] = sampler_tab.loc["night"].max() - sampler_tab.loc["night"]

    # finally, convert to int
    sampler_tab = sampler_tab.round()
    sampler_tab = sampler_tab.astype("int32", copy=True)
    over_tab = over_tab.round()
    over_tab = over_tab.astype("int32", copy=True)

    # print some stats

    if verbose > 0:
        # Output basic summary information

        print("\n====================================================")
        print("Summary statistics from stratified_sampler")
        print("====================================================")

        print("\nsampler_tab")
        sampler_tab_show = sampler_tab.copy()
        sampler_tab_show.loc[:, 'total'] = sampler_tab_show.sum(axis=1)
        sampler_tab_show.loc['total', :] = sampler_tab_show.sum(axis=0)
        sampler_tab_show = sampler_tab_show.astype("int32")
        print(sampler_tab_show)

        print("\nover_tab")
        over_tab_show = over_tab.copy()
        over_tab_show.loc[:, 'total'] = over_tab_show.sum(axis=1)
        over_tab_show.loc['total', :] = over_tab_show.sum(axis=0)
        over_tab_show = over_tab_show.astype("int32")
        print(over_tab_show)

        print("\nsampler_tab + over_tab")
        comb_tab_show = over_tab_show + sampler_tab_show
        print(comb_tab_show)

        print("\nfraction oversampled (%)")
        percover_tab_show = over_tab_show / (over_tab_show + sampler_tab_show) * 100
        print(percover_tab_show)

    if verbose > 1:
        # Output some extra information
        print("\ncross_total")
        print(cross_total)

        print("\ncross_avail")
        print(cross_avail)

        print("\n_dist_weather_classes")
        print(_dist_weather_classes)

    # return sampler table and oversampler table
    return sampler_tab, over_tab


def pandas_to_bddjson(df, dest_path):
    ### Prepare data frame for json output
    # revolve formatting back to BDD original formatting
    df["timestamp"] = 1000
    df.name = df.name.apply(os.path.basename)
    df["attributes"] = df.apply(lambda row: {'weather':row['weather'], 'scene':row['scene'], 'timeofday':row['timeofday']}, axis=1)  # This gives warning: another try: cur_file["attributes"] = [{'weather': we, 'scene': sc, 'timeofday': tod} for we, sc, tod in zip(cur_file.weather, cur_file.scene, cur_file.timeofday)]
    df = df[["attributes", "labels", "name", "timestamp"]]
    # write json file to hdd
    df.to_json(path_or_buf=dest_path)


cfg_name = '/home/till/projects/night-drive/config_bdd_make_datasets.json'
cfg = bdd.GetConfig(cfg_name)

# Load full BDD dataset
data = bdd.BaseDataset(cfg)
data = data.data

# cross-tabulation of available samples in space time x weather
crosstab_total = pd.crosstab(data['timeofday'], data['weather'])
crosstab_total = crosstab_total.reindex(sorted(crosstab_total.columns), axis=1)  # columns need to be in same order as sampler_table

cfg = bdd.GetConfig(cfg_name)
# initialize sampler_table and over_table dict to collect all the outputs
sampler_table = {}
over_table = {}
# initialize cross tabulation of remaining available images to choose from
crosstab_avail = crosstab_total.copy()

# first, we get the test set
sampler_table["test"], over_table["test"] = stratified_sampler(crosstab_total, crosstab_avail, cfg.sampler_dict["test"])
# update the numbers of remaining available images
crosstab_avail = crosstab_avail - sampler_table["test"]


# second, we get the valid set
sampler_table["valid"], over_table["valid"] = stratified_sampler(crosstab_total, crosstab_avail, cfg.sampler_dict["valid"])
# update the numbers of remaining available images
crosstab_avail = crosstab_avail - sampler_table["valid"]


# third, we get all the different train sets; note we are not updating the remaining available samples here
sampler_table["train_A"], over_table["train_A"] = stratified_sampler(crosstab_total, crosstab_avail, cfg.sampler_dict["train_A"])
sampler_table["train_B"], over_table["train_B"] = stratified_sampler(crosstab_total, crosstab_avail, cfg.sampler_dict["train_B"])
sampler_table["train_C"], over_table["train_C"] = stratified_sampler(crosstab_total, crosstab_avail, cfg.sampler_dict["train_C"])


train_sets = [k for k in cfg.sampler_dict.keys() if 'train' in k]
sets = ["set_"+k[-1] for k in train_sets]


# the train-dev sets are taken as subsets of the train sets
for train_set in train_sets:
    train_dev_set = "train_dev_" + train_set[-1]
    sampler_table[train_dev_set] = sampler_table[train_set] * cfg.train_dev_n / cfg.sampler_dict[train_set]["n"]
    sampler_table[train_set] = sampler_table[train_set] - sampler_table[train_dev_set]
    over_table[train_dev_set] = over_table[train_set] * cfg.train_dev_n / cfg.sampler_dict[train_set]["n"]
    over_table[train_set] = over_table[train_set] - over_table[train_dev_set]


# add a column to the dataframe indicating split association (train, train-dev, dev, and test set)
data["set_all"] = 'unassigned'
for s in sets:
    data[s] = 'unassigned'  # initialize with all 'unassigned'
    data[s+"_n_over"] = 0
# shuffle data and reset index
data = data.sample(frac=1.0, random_state=123).reset_index(drop=True)
# set seed for numpy
np.random.seed(123)

# stratified random sampling of indices for split sets based on sampler_table
for split, table in sampler_table.items():  # for each split
    for tod in table.index:  # for each timeofday
        for wc in table.columns:  # for each weather condition
            n_samples = int(table.loc[tod,wc])
            if n_samples > 0:
                if split in ["valid", "test"]:
                    class_idx = data[(data.set_all.eq('unassigned') & data.timeofday.eq(tod) & data.weather.eq(wc))].index
                    idx = class_idx[0:n_samples]  # np.random.choice(self.data.index[class_bool].values, size=n_samples, replace=False)
                    data.loc[idx,["set_all", *sets]] = split
                else:
                    cur_set = "set_"+split[-1]
                    if "train" in split and "train_dev" not in split:
                        # try to use data already in another train set
                        idx = data[(data.set_all.str.contains('train') & data.timeofday.eq(tod) & data.weather.eq(wc))].index
                        n = idx.size
                        if n >= n_samples:
                            idx = idx[0:n_samples]
                        else:
                            idx_add = data[(data.set_all.eq('unassigned') & data.timeofday.eq(tod) & data.weather.eq(wc))].index
                            idx = np.hstack([idx, idx_add[0:n_samples - n]] )
                        data.loc[idx,["set_all", cur_set]] = "train"
                    elif "train_dev" in split:
                        # try to use data already in another train dev set
                        idx = data[(data.set_all.str.contains('train_dev') & data.timeofday.eq(tod) & data.weather.eq(wc))].index
                        n = idx.size
                        if n >= n_samples:
                            idx = idx[0:n_samples]
                        else:
                            # try to use data already in another train set
                            idx_add = data[(data.set_all.str.contains('train') & ~data[cur_set].str.contains('train') & data.timeofday.eq(tod) & data.weather.eq(wc))].index
                            n_add = idx_add.size
                            idx = np.hstack([idx, idx_add[0:np.minimum(n_samples - n, n_add)]] )
                            n = idx.size
                            if n < n_samples:
                                idx_add = data[(data.set_all.eq('unassigned') & data.timeofday.eq(tod) & data.weather.eq(wc))].index
                                n_add = idx_add.size
                                idx = np.hstack([idx, idx_add[0:np.minimum(n_samples - n, n_add)]] )
                        data.loc[idx,["set_all", cur_set]] = "train_dev"
                    # store number of over-samples
                    idx_over = np.random.choice(idx, over_table[split].loc[tod,wc].astype("int"))
                    idx_uni, counts = np.unique(idx_over, return_counts=True)
                    data.loc[idx_uni, cur_set+"_n_over"] = counts


# create a useful info_dict containing info about each split element of sampler dict
info_dict = {}
info_dict["splits"] = sampler_table.keys()
for split in info_dict["splits"]:
    info_dict[split] = {}
    # set association
    if split in ["test", "valid"]:
        info_dict[split]["set"] = "set_all"
    else:
        info_dict[split]["set"] = "set_" + split[-1]
    # part name
    if split in ["test", "valid"]:
        info_dict[split]["split"] = split
    else:
        info_dict[split]["split"] = split[:-2]
    # destination path
    if cfg.do_make_dirs:  # create a separate dir for each split
        info_dict[split]["destination_path"] = os.path.join(cfg.destination_path, split)
    else:  # create all files in the same dir
        info_dict[split]["destination_path"] = cfg.destination_path
    # destination file names
    info_dict[split]["destination_json_filename"] = cfg.destination_filename_stem + split + ".json"
    info_dict[split]["destination_json_over_filename"] = cfg.destination_filename_stem + split + "_over" + ".json"
    # destination file path
    info_dict[split]["destination_json_filepath"] = os.path.join(info_dict[split]["destination_path"], info_dict[split]["destination_json_filename"])
    info_dict[split]["destination_json_over_filepath"] = os.path.join(info_dict[split]["destination_path"], info_dict[split]["destination_json_over_filename"])
