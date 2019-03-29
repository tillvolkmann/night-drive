"""
The script requires a json file in BDD format.

Example call:
cd /home/SharedFolder/git/tillvolkmann/night-drive/classifier_timeofday/
python3 ./timeofday_eval.py

"""
project_root = "/home/SharedFolder/CurrentDatasets/bdd100k/"
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.models as models
import torchvision.transforms as transforms
import os, sys
sys.path.append("/home/SharedFolder/git/tillvolkmann/night-drive/")
sys.path.append("/home/SharedFolder/git/tillvolkmann/night-drive/datasets")
sys.path.append("/home/SharedFolder/git/tillvolkmann/night-drive/night-drive/")
sys.path.append("/home/SharedFolder/git/tillvolkmann/night-drive/night-drive/datasets")
import datasets.bdd.BDDTimeOfDayDataset as bdd


# setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device}")

# setting data set root dir
root_dir = "/home/SharedFolder/CurrentDatasets/bdd100k"
which_split = "bddvalid_converted"  # "bddvalid_converted"
results_json_file = os.path.join(root_dir, which_split+"-timeofday-results.json")
which_ganmodel="v032_e14"
batch_size = 32 # 128  # 256
num_workers = 2 # 2 # 16

# data transforms
transform = {
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) # ImageNet
}

# create data sets
dataset = bdd.BDDTimeOfDayDataset(
    root_dir,
    split=which_split,
    transform=transform["test"],
    with_labels=True,
    dropcls=['dawn/dusk', 'undefined'],
    output_filenames=True,
    which_ganmodel=which_ganmodel)


# data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# create model
net = models.resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 1)

# send model to device
net.to(device)

# load weights
net.load_state_dict(torch.load("./models/resnet18_timeofday_daynight_classifier_best.pth")["model_state_dict"])

# set model to eval mode
net.eval()

with torch.no_grad():
    # initilaize results data frame
    timeofday_predicted = pd.DataFrame(columns=["name", "label", "prediction"])
    # Loop through data
    for i, data in enumerate(dataloader, 0):
        print(f"Processing batch {i + 1} of {(len(dataset) // batch_size) + 1}...")
        inputs, labels, filenames = data
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = nnf.sigmoid(outputs) # for probabilities (binary classification)
        # pd.DataFrame({"name": list(filenames), "predicted_timeofday": list(outputs.detach().cpu().numpy().squeeze())})
        timeofday_predicted = pd.concat([timeofday_predicted, pd.DataFrame({
            "name": list(filenames),
            "label": labels.detach().cpu().numpy().squeeze(),
            "prediction": list(outputs.detach().cpu().numpy().squeeze())})], sort=True, ignore_index=True)
    # Write results to json file
    timeofday_predicted = timeofday_predicted.reset_index(drop=True)
    timeofday_predicted.to_json(results_json_file)

