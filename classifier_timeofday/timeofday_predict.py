"""
The script requires a json file in BDD format.

Example call:
python3 /home/SharedFolder/git/tillvolkmann/night-drive/classifier_timeofday/timeofday_predict.py
cd /home/SharedFolder/git/tillvolkmann/night-drive/classifier_timeofday/
python3 ./timeofday_predict.py

"""
project_root = "/home/SharedFolder/CurrentDatasets/bdd100k/"
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.models as models
import torchvision.transforms as transforms
import sys
sys.path.append("/home/SharedFolder/git/tillvolkmann/night-drive")
sys.path.append("/home/SharedFolder/git/tillvolkmann/night-drive/datasets")
import datasets.bdd.BDDTimeOfDayDataset as bdd


# setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device}")

# setting data set root dir
root_dir = "/home/SharedFolder/CurrentDatasets/bdd100k/"
which_split = "bddvalid_converted"
results_json_file = which_split+"-timeofday-results.json"
batch_size = 256

# data transforms
transform = {
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]) # ImageNet
}

# create data sets
dataset = bdd.BDDTimeOfDayDataset(root_dir, split=which_split, transform=transform["test"], with_labels = False)

# data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 16)

# create model
net = models.resnet18(pretrained = True)
net.fc = nn.Linear(net.fc.in_features, 1)

# send model to device
net.to(device)

# load weights
net.load_state_dict(torch.load("./models/resnet18_timeofday_daynight_classifier_best.pth")["model_state_dict"])

# set model to eval mode
net.eval()

with torch.no_grad():
    timeofday_predicted = pd.DataFrame(columns=["name", "predicted_timeofday"])
    for i, data in enumerate(dataloader, 0):
        print(f"Processing batch {i + 1} of {(len(dataset) // batch_size) + 1}...")
        inputs, filenames = data
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = nnf.sigmoid(outputs) # for probabilities (binary classification)
        # pd.DataFrame({"name": list(filenames), "predicted_timeofday": list(outputs.detach().cpu().numpy().squeeze())})
        timeofday_predicted = pd.concat([timeofday_predicted, pd.DataFrame({"name": list(filenames),
                            "predicted_timeofday": list(outputs.detach().cpu().numpy().squeeze())})], ignore_index=True)
    timeofday_predicted = timeofday_predicted.reset_index(drop=True)
    timeofday_predicted.to_json(results_json_file)

