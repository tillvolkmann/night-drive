#!/bin/bash

#python weather_train.py \
#    --path_train_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_A_over.json \
#    --path_train_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_A \
#    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_dev_A_over.json \
#    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_dev_A \
#    > log_train_A_over.txt

#python weather_train.py \
#    --path_train_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_B_over.json \
#    --path_train_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_B \
#    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_dev_B_over.json \
#    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_dev_B \
#    > log_train_B_over.txt

#python weather_train.py \
#    --path_train_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_C_over.json \
#    --path_train_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_C \
#    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_dev_C_over.json \
#    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_dev_C \
#    > log_train_C_over.txt

#python weather_train.py \
#    --path_train_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/annotations/bdd100k_sorted_train_A_ganaug_025_over.json \
#    --path_train_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/train_A_ganaug_025 \
#    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/annotations/bdd100k_sorted_train_dev_A_ganaug_025_over.json \
#    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/train_dev_A_ganaug_025 \
#    > log_train_A_ganaug_025_over.txt

#python weather_train.py \
#    --path_train_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/annotations/bdd100k_sorted_train_A_ganaug_050_over.json \
#    --path_train_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/train_A_ganaug_050 \
#    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/annotations/bdd100k_sorted_train_dev_A_ganaug_050_over.json \
#    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/train_dev_A_ganaug_050 \
#    > log_train_A_ganaug_050_over.txt

#python weather_train.py \
#    --path_train_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/annotations/bdd100k_sorted_train_B_ganaug_025_over.json \
#    --path_train_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/train_B_ganaug_025 \
#    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/annotations/bdd100k_sorted_train_dev_B_ganaug_025_over.json \
#    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_augmented_v032_e14/train_dev_B_ganaug_025 \
#    > log_train_B_ganaug_025_over.txt

python weather_train.py \
    --path_train_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_A_ganaug_050_over_augonlyasbase.json \
    --path_train_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_A \
    --path_valid_json /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_dev_A_ganaug_050_over_augonlyasbase.json \
    --path_valid_images /home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_dev_A \
    > log_train_A_ganaug_050_over_augonlyasbase.txt
