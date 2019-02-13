import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import datasets.bdd.BDDWeatherDataset as bdd

if __name__ == '__main__':

    # setting device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # setting data set root dir
    root_dir = "/home/SharedFolder/CurrentDatasets/bdd100k"

    # data transforms
    transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]) # ImageNet
    }

    # create data set
    ds_train = bdd.BDDWeatherDataset(root_dir, split = "bddtrain", transform = transform["train"], force_num = 1000)

    # create model
    net = models.resnet50(pretrained = True)
    net.fc = nn.Linear(net.fc.in_features, ds_train._get_num_classes())

    # data loader
    ld_train = torch.utils.data.DataLoader(ds_train, batch_size = 64, shuffle = True, num_workers = 16)

    # send model to device
    net.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(net.parameters())

    # number of epochs
    num_epochs = 100

    # log every n mini batches
    log_step = 1

    tic = time.time()

    for epoch in range(num_epochs):  # loop over the data set multiple times

        running_loss = 0.0

        for i, data in enumerate(ld_train, 0):

            # get the inputs
            inputs, targets = data

            # send inputs to device
            inputs, targets = inputs.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = net(inputs)

            # calculate loss
            loss = criterion(outputs, targets)

            # calculate gradients
            loss.backward()

            # learn
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % log_step == 0: # print every 200 mini-batches
                eta_hours = (((time.time() - tic) / (epoch + 1)) * (num_epochs - (epoch + 1))) / 3600.0
                print(f"Epoch {epoch + 1}, Batch {i + 1}: Training Loss = {(running_loss / log_step):.3f} ETA = {eta_hours:.2f}h")
                running_loss = 0.0
