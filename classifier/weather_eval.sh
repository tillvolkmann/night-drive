#!/bin/bash

## WITHOUT CLOUDY #

# train_A_over

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_dev_A_over.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_dev_A \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

# train_B_over

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_dev_B_over.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_dev_B \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

# train_C_over

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_C_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_dev_C_over.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_dev_C \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_C_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_C_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_C_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_C_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_C_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_C_over \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

# train_A_over_ganaug_025

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/annotations/bdd100k_sorted_train_dev_A_ganaug_025_over.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/train_dev_A_ganaug_025 \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

# train_A_over_ganaug_050

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/annotations/bdd100k_sorted_train_dev_A_ganaug_050_over.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/train_dev_A_ganaug_050 \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

# train_B_over_ganaug_025

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/annotations/bdd100k_sorted_train_dev_B_ganaug_025_over.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/train_dev_B_ganaug_025 \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_B_over_ganaug_025 \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

# train_A_over_ganaug_050_augonlyasbase

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050_augonlyasbase \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_dev_A_ganaug_050_over_augonlyasbase.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_dev_A \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050_augonlyasbase \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050_augonlyasbase \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050_augonlyasbase \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050_augonlyasbase \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050_augonlyasbase \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_daytime.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy

python weather_eval.py \
    --dir_weights /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy/train_A_over_ganaug_050_augonlyasbase \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_test_night.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/test \
    --out_folder /home/SharedFolder/trained_models/night-drive/weather_classifier/without_cloudy
