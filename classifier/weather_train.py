import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import datasets.bdd.BDDWeatherDataset as bdd
import sklearn.metrics as metrics
import albumentations.augmentations.transforms as transforms
from albumentations import Compose
from albumentations.pytorch import ToTensor

# Logging
#import neptune
#from tensorboardX import SummaryWriter

# setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_f1_score(net, data_loader, num_batches = None):
    # f1 score: https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel
    net.eval() # disables dropout, etc.
    with torch.no_grad(): # temporarily disables gradient computation for speed-up
        accumulated_targets = []
        accumulated_outputs = []
        for i, data in enumerate(data_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            accumulated_targets.extend(targets.detach().cpu().numpy().tolist())
            accumulated_outputs.extend(np.argmax(outputs.detach().cpu().numpy().tolist(), axis = 1))
            if num_batches is not None and (i >= (num_batches - 1)):
                break
        f1_score = metrics.f1_score(accumulated_targets, accumulated_outputs, average="weighted")
    net.train()
    return f1_score


if __name__ == "__main__":

    # set data paths
    path_train_images = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_A"
    path_train_json = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_A_over.json"
    path_valid_images = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/train_dev_A"
    path_valid_json = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_train_dev_A_over.json"

    # seeds
    torch.manual_seed(123)
    np.random.seed(123)

    # setup job monitoring on Neptune
    #ctx = neptune.Context()

    # setup job monitoring on TensorBoardX
    #writer = SummaryWriter()

    # show validation loss
    calc_valid_loss = True

    # data transforms
    t_target_size = (448, 448)
    t_norm_mean = [0.485, 0.456, 0.406] # ImageNet
    t_norm_std = [0.229, 0.224, 0.225] # ImageNet
    transform = {
        "train": Compose([
            transforms.RandomBrightnessContrast(brightness_limit = 0.2, contrast_limit = 0.2, p = 0.5),
            transforms.MedianBlur(blur_limit = 3, p = 0.5),
            transforms.GaussNoise(var_limit = (1.0, 50.0), p = 0.5),
            transforms.HorizontalFlip(p = 0.5),
            transforms.RandomSizedCrop(min_max_height = (t_target_size[0] // 2, t_target_size[0]), height = t_target_size[0], width = t_target_size[1], w2h_ratio = 1.777778),
            transforms.Resize(height = t_target_size[0], width = t_target_size[1]),
            ToTensor(normalize = {"mean": t_norm_mean, "std": t_norm_std})]),
        "valid": Compose([
            transforms.Resize(height = t_target_size[0], width = t_target_size[1]),
            ToTensor(normalize = {"mean": t_norm_mean, "std": t_norm_std})])
    }

    # create data sets
    ds_train = bdd.BDDWeatherDataset(path_train_json, path_train_images, transform = transform["train"], drop_cls = ["cloudy"])
    ds_valid = bdd.BDDWeatherDataset(path_valid_json, path_valid_images, transform = transform["valid"], drop_cls = ["cloudy"])

    # data loader
    dl_batch_size = 32
    dl_num_workers_train = 10
    dl_num_workers_valid = 6
    dl_shuffle = True
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size = dl_batch_size, shuffle = dl_shuffle, num_workers = dl_num_workers_train)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size = dl_batch_size, shuffle = dl_shuffle, num_workers = dl_num_workers_valid)

    # create model
    net = models.resnet18(pretrained = True)
    net.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Needed for resolutions > 224 x 224
    net.fc = nn.Linear(net.fc.in_features, ds_train._get_num_classes())

    # send model to device
    net.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)

    # number of epochs
    num_epochs = 20

    # log every n mini batches
    log_step = 10

    # save every m epochs
    epoch_save_step = 1

    tic = time.time()

    for epoch in range(num_epochs): # loop over the data set multiple times

        running_loss_train = 0.0
        running_loss_valid = 0.0
        data_valid = iter(dl_valid)

        for i, data in enumerate(dl_train, 0):

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

            # accumulate training loss for printing
            running_loss_train += loss.item()

            # calculate validation loss
            if calc_valid_loss:
                net.eval()
                try:
                    inputs_valid, targets_valid = next(data_valid)
                except StopIteration:
                    data_valid = iter(dl_valid)
                    inputs_valid, targets_valid = next(data_valid)
                inputs_valid, targets_valid = inputs_valid.to(device), targets_valid.to(device)
                outputs_valid = net(inputs_valid)
                loss_valid = criterion(outputs_valid, targets_valid)
                running_loss_valid += loss_valid.item()
                net.train()

            # print statistics
            if (i + 1) % log_step == 0: # print every log_step mini-batches
                num_batches_done = epoch * len(dl_train) + i + 1
                num_batches_remaining = num_epochs * len(dl_train) - num_batches_done
                eta = (((time.time() - tic) / num_batches_done) * num_batches_remaining) / 3600
                if calc_valid_loss:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(dl_train)}: Train Loss = {(running_loss_train / log_step):.3f} "
                          f"Valid Loss = {(running_loss_valid / log_step):.3f} ETA = {eta:.2f}h")
                else:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(dl_train)}: Train Loss = {(running_loss_train / log_step):.3f} "
                          f"ETA = {eta:.2f}h")
                running_loss_train = 0.0
                running_loss_valid = 0.0

            # send to Neptune and TensorBoardX
            #ctx.channel_send('train_loss', i, loss.data.cpu().numpy())
            #writer.add_scalar('train/loss', loss.data.cpu().numpy(), i)
            #if calc_valid_loss:
            #    ctx.channel_send('valid_loss', i, loss_valid.data.cpu().numpy())
            #    writer.add_scalar('valid/loss', loss_valid.data.cpu().numpy(), i)

        # save model
        if ((epoch + 1) % epoch_save_step == 0) or (epoch + 1) == num_epochs: # save every epoch_save_step epochs
            torch.save({
                "epochs": epoch + 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": loss.data.cpu().numpy(),
            }, f"./resnet18_weather_classifier_{path_train_json.split(os.sep)[-1].split('_sorted')[-1].split('.json')[0]}_epoch_{epoch + 1}.pth")

    # close TensorBoardX
    #writer.close()
