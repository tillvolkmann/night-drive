import os
import re
import json
import pandas as pd

input_file = "./bdd/bdd-test-timeofday-results.json"
prob_thresh = 0.5
patch_json = ""

if not os.path.exists(patch_json):
    inference_timeofday = pd.read_json(input_file)
    inference_timeofday = inference_timeofday.reset_index(drop = True)
    output_list = []
    num_day = 0
    num_night = 0
    num_undef = 0
    for index, row in inference_timeofday.iterrows():
        if row["predicted_timeofday"] >= prob_thresh:
            timeofday = "night"
            num_night += 1
        elif row["predicted_timeofday"] <= (1 - prob_thresh):
            timeofday = "daytime"
            num_day += 1
        else:
            timeofday = "undefined"
            num_undef += 1
        frame = {
            "name": row["name"].split(os.sep)[-1],
            "attributes": {
                "weather": "undefined",
                "scene": "undefined",
                "timeofday": timeofday
            },
            "timestamp": 0
        }
        output_list.append(frame)
    output_file = input_file.split(".json")[0] + "_thresh_" + str(prob_thresh) + "_" + str(num_day) + "-" + str(num_night) \
                  + "-" + str(num_undef) + ".json"
    with open(output_file, "w") as f:
        f.write(json.dumps(output_list, indent = 4))
else:
    with open(patch_json) as f:
        label_file_to_patch = f.read()
    inference_timeofday = pd.read_json(input_file)
    inference_timeofday = inference_timeofday.reset_index(drop = True)
    output_list = []
    num_day = 0
    num_night = 0
    num_undef = 0
    for index, row in inference_timeofday.iterrows():
        old_pattern = "\"name\": \"" + row["name"].split(os.sep)[-1] \
                  + "\",\n        \"attributes\": {\n            \"weather\": \"undefined\",\n            \"scene\": \"undefined\",\n            \"timeofday\": \"undefined\""
        if row["predicted_timeofday"] >= prob_thresh:
            new_pattern = "\"name\": \"" + row["name"].split(os.sep)[-1] \
                  + "\",\n        \"attributes\": {\n            \"weather\": \"undefined\",\n            \"scene\": \"undefined\",\n            \"timeofday\": \"night\""
            num_night += 1
        elif row["predicted_timeofday"] <= (1 - prob_thresh):
            new_pattern = "\"name\": \"" + row["name"].split(os.sep)[-1] \
                  + "\",\n        \"attributes\": {\n            \"weather\": \"undefined\",\n            \"scene\": \"undefined\",\n            \"timeofday\": \"daytime\""
            num_day += 1
        else:
            new_pattern = "\"name\": \"" + row["name"].split(os.sep)[-1] \
                  + "\",\n        \"attributes\": {\n            \"weather\": \"undefined\",\n            \"scene\": \"undefined\",\n            \"timeofday\": \"undefined\""
            num_undef += 1
        label_file_to_patch = re.sub(old_pattern, new_pattern, label_file_to_patch)
    output_file = patch_json.split(".json")[0] + "_thresh_" + str(prob_thresh) + "_" + str(num_day) + "-" + str(num_night) \
                  + "-" + str(num_undef) + ".json"
    with open(output_file, "w") as f:
        f.write(label_file_to_patch)
