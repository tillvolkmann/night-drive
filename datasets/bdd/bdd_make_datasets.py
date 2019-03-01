import os
import numpy as np
import pandas as pd
from PIL import Image
import datasets.bdd.BDDDataSets as bdd


def stratified_sampler(crstab, config):
    if config.sampler_dict['class_min'] is not None:
        sampler_table.loc[:, :] = np.min(crstab, config.sampler_dict['class_min'])



# Notes:
# This setting creates
# "n":
#   int ~ number of samples
#   'max' ~ take max number of remaining samples (only applicable for train)
# "balancing":
#   "over_after"  ~ balances after selecting fraction of night, i.e. target night fraction will refer to fraction of original pics
#   "over_before" ~ balances before selecting fraction of night, i.e. target night fraction will refer to fraction of original pics
# "none" : within each, day and night, use original dist of weather classes
# "daynight" : use same dist of weather classes for day and night

# config
do_balance_timeofday_after_oversampling = True
base_data_cap_for_balance = 10000
do_oversample_physically = True
origin_path = ''
destination_path = ''
splits = ['train', 'train_dev', 'valid', 'test']

cfg_name = '/home/till/projects/night-drive/config_bdd_make_datasets.json'
cfg = bdd.GetConfig(cfg_name)


# Load full BDD dataset
data = bdd.BaseDataset(cfg)

# Balance base data set if requested
if base_data_cap_for_balance is not None:
    pass

if do_balance_timeofday_after_oversampling:
    pass

# definitely sample according to the original data distributino
# with the only exception of more cloudy at night

# first sample 100% day

# then 50% night

# then subset the rest

# Make json files
# Test and valid sets


# Copy images into destination folders

if do_oversample_physically:
    pass


