import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import datasets.bdd.BDDWeatherDataset as bdd
import sklearn.metrics as metrics

# Logging
#import neptune
#from tensorboardX import SummaryWriter


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
    path_valid_images = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/valid"
    path_valid_json = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_coco/annotations/bdd100k_sorted_valid.json"

    # seeds
    torch.manual_seed(123)
    np.random.seed(123)

    # setup job monitoring on Neptune
    #ctx = neptune.Context()

    # setup job monitoring on TensorBoardX
    #writer = SummaryWriter()

    # setting device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # show validation loss
    calc_valid_loss = False

    # data transforms
    #t_target_size = (224, 224) # default resolution
    t_target_size = (720, 1280) # full bdd resolution
    t_norm_mean = [0.485, 0.456, 0.406]
    t_norm_std = [0.229, 0.224, 0.225]
    transform = {
        "train": transforms.Compose([
            transforms.Resize(t_target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=t_norm_mean, std=t_norm_std)]),  # ImageNet
        "valid": transforms.Compose([
            transforms.Resize(t_target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=t_norm_mean, std=t_norm_std)])  # ImageNet
    }

    # create data sets
    ds_train = bdd.BDDWeatherDataset(path_train_json, path_train_images, transform = transform["train"])
    ds_valid = bdd.BDDWeatherDataset(path_valid_json, path_valid_images, transform = transform["valid"])

    # data loader
    dl_batch_size = 6
    dl_num_workers = 8
    dl_shuffle = True
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size = dl_batch_size, shuffle = dl_shuffle, num_workers = dl_num_workers)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size = dl_batch_size, shuffle = dl_shuffle, num_workers = dl_num_workers)

    # create model
    net = models.resnet50(pretrained = True)
    net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    net.fc = nn.Linear(net.fc.in_features, ds_train._get_num_classes())

    # send model to device
    net.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = 0.0001, weight_decay = 0.001)

    # number of epochs
    num_epochs = 50

    # log every n mini batches
    log_step = 1

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

        # f1-score on training set, validation set
        #f1_train = evaluate_f1_score(net, dl_train)
        #f1_valid = evaluate_f1_score(net, dl_valid)
        #print(f"F1-Score after {epoch + 1} epochs: Training = {f1_train:.2f} Validation = {f1_valid:.2f}")

        # save model
        if ((epoch + 1) % epoch_save_step == 0) or (epoch + 1) == num_epochs: # save every epoch_save_step epochs
            torch.save({
                "epochs": epoch + 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": loss.data.cpu().numpy(),
                "dataset": path_train_json,
                #"train_f1": f1_train,
                #"valid_f1": f1_valid
            }, f"./resnet50_weather_classifier_epoch_{epoch + 1}.pth")

    # close TensorBoardX
    #writer.close()
