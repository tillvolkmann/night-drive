import os
import cv2
import glob
import json
import time
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool

''' BDD label format
- name: string
- url: string
- videoName: string (optional)
- attributes:
    - weather: "rainy|snowy|clear|overcast|undefined|partly cloudy|foggy"
    - scene: "tunnel|residential|parking lot|undefined|city street|gas stations|highway|"
    - timeofday: "daytime|night|dawn/dusk|undefined"
- intrinsics
    - focal: [x, y]
    - center: [x, y]
    - nearClip:
- extrinsics
    - location
    - rotation
- timestamp: int64 (epoch time ms)
- frameIndex: int (optional, frame index in this video)
- labels [ ]:
    - id: int32
    - category: string (classification)
    - manualShape: boolean (whether the shape of the label is created or modified manually)
    - manualAttributes: boolean (whether the attribute of the label is created or modified manually)
    - attributes:
        - occluded: boolean
        - truncated: boolean
        - trafficLightColor: "red|green|yellow|none"
        - areaType: "direct | alternative" (for driving area)
        - laneDirection: "parallel|vertical" (for lanes)
        - laneStyle: "solid | dashed" (for lanes)
        - laneTypes: (for lanes)
    - box2d:
       - x1: float
       - y1: float
       - x2: float
       - y2: float
   - box3d:
       - alpha: (observation angle if there is a 2D view)
       - orientation: (3D orientation of the bounding box, used for 3D point cloud annotation)
       - location: (3D point, x, y, z, center of the box)
       - dimension: (3D point, height, width, length)
   - poly2d: an array of objects, with the structure
       - vertices: [][]float (list of 2-tuples [x, y])
       - types: string (each character corresponds to the type of the vertex with the same index in vertices. ‘L’ for vertex and ‘C’ for control point of a bezier curve.
       - closed: boolean (closed for polygon and otherwise for path)
'''

########################################################################################################################

dict_mapillary_bdd = {
    "Person": "person",
    "Bicyclist": "rider",
    "Motorcyclist": "rider",
    "Other Rider": "rider",
    "Traffic Light": "traffic light",
    "Traffic Sign (Front)": "traffic sign", # skipping Traffic Sign (Back)
    "Bicycle": "bike",
    "Bus": "bus",
    "Car": "car",
    "Caravan": "bus",
    "Motorcycle": "motor",
    "On Rails": "train",
    "Truck": "truck"
}
dataset_rootpath = '/home/SharedFolder/CurrentDatasets/mapillary-vistas-dataset_public_v1.1'
#dataset_subpaths = ['validation', 'testing', 'training']
dataset_subpaths = ['training']
dataset_colcodes = 'config.json'

########################################################################################################################

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def generate_labels(objects):
    counter = 0
    labels = []
    for panoptic_bgr_type, coords in objects.items():
        rect = cv2.boundingRect(np.array(coords))
        y0 = rect[0]
        x0 = rect[1]
        y1 = rect[0] + rect[2]
        x1 = rect[1] + rect[3]
        mapillary_object_type = panoptic_bgr_type.split('_')[1]
        bdd_object_type = dict_mapillary_bdd[mapillary_object_type]
        label = {
            "category": bdd_object_type,
            "attributes": {
                "occluded": False,
                "truncated": False,
                "trafficLightColor": "none"
            },
            "manualShape": True,
            "manualAttributes": True,
            "box2d": {
                "x1": x0,
                "y1": y0,
                "x2": x1,
                "y2": y1
            },
            "id": counter
        }
        labels.append(label)
        counter += 1
    return labels

def extract_objects(lbl_image, pan_image):
    objects = {}
    for mapillary_object_type in dict_mapillary_bdd:
        bgr_code = df_mapillary[df_mapillary["readable"] == mapillary_object_type]["color"].to_list()[0]
        mask = cv2.inRange(lbl_image, np.array(bgr_code), np.array(bgr_code))
        coords = cv2.findNonZero(mask)
        if coords is None:
            coords = []
        for coord in coords:
            y = coord[0][1]
            x = coord[0][0]
            panoptic_bgr_type = str(pan_image[y][x][0]) + str(pan_image[y][x][1]) + str(pan_image[y][x][2])
            panoptic_bgr_type = panoptic_bgr_type + '_' + mapillary_object_type
            if panoptic_bgr_type in objects:
                objects[panoptic_bgr_type].append([[y, x]])
            else:
                objects[panoptic_bgr_type] = [[[y, x]]]
    return objects

def process_frame(valid_paths):
    _tic = time.time()
    num_curr = valid_paths[0]
    num_total = valid_paths[1]
    image_path = valid_paths[2]
    label_path = valid_paths[3]
    panop_path = valid_paths[4]
    # double check
    label_short_name = label_path.split(os.sep)[-1].split('.png')[0]
    assert image_path.split(os.sep)[-1].split('.jpg')[0] == label_short_name
    assert label_short_name == panop_path.split(os.sep)[-1].split('.png')[0]
    lbl_image = cv2.imread(label_path)
    pan_image = cv2.imread(panop_path)
    # find objects and generate labels
    objects = extract_objects(lbl_image, pan_image)
    frame = {
        "name": image_path.split(os.sep)[-1],
        "attributes": {
            "weather": "undefined",
            "scene": "undefined",
            "timeofday": "undefined"
        },
        "timestamp": 0,
        "labels": generate_labels(objects)
    }
    # lists claim to be thread-safe...
    output_list.append(frame)
    _toc = time.time()
    print(f"+++ Processed image {num_curr} of {num_total} in {_toc - _tic :.1f}s...")

########################################################################################################################

tic = time.time()
with open(os.path.join(dataset_rootpath, dataset_colcodes)) as f:
    data = json.load(f)
df_mapillary = pd.DataFrame(data['labels'])
df_mapillary.color = df_mapillary.color.apply(lambda x: x[::-1])
output_list = []

for dataset_subpath in dataset_subpaths:
    print('+ Opening ' + dataset_subpath + ' set...')
    # find valid image triples
    print('++ Finding valid image triples...')
    image_paths = glob.glob(str(os.path.join(dataset_rootpath, dataset_subpath, 'images')) + '/**/*.jpg', recursive = True)
    label_paths = glob.glob(str(os.path.join(dataset_rootpath, dataset_subpath, 'labels')) + '/**/*.png', recursive = True)
    panop_paths = glob.glob(str(os.path.join(dataset_rootpath, dataset_subpath, 'panoptic')) + '/**/*.png', recursive = True)
    images = [imagePath.split(os.sep)[-1].split('.jpg')[0] for imagePath in image_paths]
    labels = [labelPath.split(os.sep)[-1].split('.png')[0] for labelPath in label_paths]
    panops = [panopPath.split(os.sep)[-1].split('.png')[0] for panopPath in panop_paths]
    valids = intersection(intersection(images, labels), panops)
    valid_image_paths = sorted([imagePath for imagePath in image_paths if imagePath.split(os.sep)[-1].split('.jpg')[0] in valids])
    valid_label_paths = sorted([labelPath for labelPath in label_paths if labelPath.split(os.sep)[-1].split('.png')[0] in valids])
    valid_panop_paths = sorted([panopPath for panopPath in panop_paths if panopPath.split(os.sep)[-1].split('.png')[0] in valids])
    assert len(valid_image_paths) == len(valid_label_paths)
    assert len(valid_label_paths) == len(valid_panop_paths)
    valid_paths = list(zip(
        list(range(1, len(valid_label_paths) + 1)),
        [len(valid_label_paths) for i in range(len(valid_label_paths))],
        valid_image_paths, valid_label_paths, valid_panop_paths
    ))

    # iterate over images in dataset
    output_list.clear()
    pool = ThreadPool(1)
    pool.map(process_frame, valid_paths)
    pool.close()
    pool.join()

    # write results
    output_file = dataset_rootpath.split(os.sep)[-1] + "_" + dataset_subpath + ".json"
    with open(os.path.join(dataset_rootpath, output_file), "w") as f:
        f.write(json.dumps(output_list, indent = 4))

toc = time.time()
print('Done in ' + str(toc - tic) + ' seconds.')