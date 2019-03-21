import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from weather_classifier import weather_classifier

# Logging
#import neptune
#from tensorboardX import SummaryWriter

# setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    # create classifer, datasets and dataloader
    net, dl_train, dl_valid, _ = weather_classifier(path_train_json = path_train_json,
                                                    path_train_images = path_train_images,
                                                    path_valid_json = path_valid_json,
                                                    path_valid_images = path_valid_images)

    # send model to device
    net.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = 0.000005)

    # number of epochs
    num_epochs = 40

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
            inputs, targets, _ = data

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
                    inputs_valid, targets_valid, _ = next(data_valid)
                except StopIteration:
                    data_valid = iter(dl_valid)
                    inputs_valid, targets_valid, _ = next(data_valid)
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
                "dataset": path_train_json,
            }, f"./resnet18_weather_classifier_{path_train_json.split(os.sep)[-1].split('.json')[0]}_epoch_{epoch + 1}.pth")

    # close TensorBoardX
    #writer.close()
