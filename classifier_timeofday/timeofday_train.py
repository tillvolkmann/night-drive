import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import datasets.bdd.BDDTimeOfDayDataset as bdd
from classification_eval import evaluate_timeofday

# Logging
#import neptune
#from tensorboardX import SummaryWriter


if __name__ == '__main__':

    # seeds
    torch.manual_seed(123)
    np.random.seed(123)

    # setup job monitoring on Neptune
    #ctx = neptune.Context()

    # setup job monitoring on TensorBoardX
    #writer = SummaryWriter()

    # setting device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # setting data set root dir
    root_dir = "/home/SharedFolder/CurrentDatasets/bdd100k"

    # show validation loss
    calc_valid_loss = False

    # data transforms
    transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]), # ImageNet
        "valid": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]) # ImageNet
    }

    # create data sets
    ds_train = bdd.BDDTimeOfDayDataset(root_dir, split = "bddtrain", transform = transform["train"], dropcls = ["dawn/dusk", "undefined"], force_num = 27971)
    ds_valid = bdd.BDDTimeOfDayDataset(root_dir, split = "bddvalid", transform = transform["valid"], dropcls = ["dawn/dusk", "undefined"])

    # data loader
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size = 256, shuffle = True, num_workers = 8)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size = 256, shuffle = True, num_workers = 8)

    # create model
    net = models.resnet18(pretrained = True)
    net.fc = nn.Linear(net.fc.in_features, 1) # for binary classfication

    # send model to device
    net.to(device)

    # loss function
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = 0.0001, weight_decay = 0.001)

    # number of epochs
    num_epochs = 25

    # log every n mini batches
    log_step = 1

    # track best f1-score
    best_f1_valid = -1

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
            loss = criterion(outputs.squeeze(), targets.float())

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
                    print(f"Epoch {epoch + 1}, Batch {i + 1}: Train Loss = {(running_loss_train / log_step):.3f} "
                          f"Valid Loss = {(running_loss_valid / log_step):.3f} ETA = {eta:.2f}h")
                else:
                    print(f"Epoch {epoch + 1}, Batch {i + 1}: Train Loss = {(running_loss_train / log_step):.3f} "
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
        f1_train = evaluate_timeofday(net, dl_train)
        f1_valid = evaluate_timeofday(net, dl_valid)
        print(f"F1-Score after {epoch + 1} epochs: Training = {f1_train:.2f} Validation = {f1_valid:.2f}")
        # save model if best so far
        if f1_valid > best_f1_valid:
            print("Snapshotting best model...")
            best_f1_valid = f1_valid
            torch.save({
                'epochs': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': loss.data.cpu().numpy(),
                'train_f1': f1_train,
                'valid_f1': f1_valid
            }, './models/resnet18_timeofday_daynight_classifier_best.pth')

    # save final model
    torch.save({
        'epochs': num_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': loss.data.cpu().numpy(),
        'train_f1': f1_train,
        'valid_f1': f1_valid
    }, './models/resnet18_timeofday_daynight_classifier_final.pth')

    # close TensorBoardX
    #writer.close()
