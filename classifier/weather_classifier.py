import time
import torch
import neptune
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import datasets.bdd.BDDWeatherDataset as bdd
import sklearn.metrics as metrics
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
            accumulated_outputs.extend(np.argmax(outputs.detach().cpu().numpy().tolist(), axis=1))
            if num_batches is not None and (i >= (num_batches - 1)):
                break
        f1_score = metrics.f1_score(accumulated_targets, accumulated_outputs, average="weighted")
    net.train()
    return f1_score

if __name__ == '__main__':

    # seeds
    torch.manual_seed(123)
    np.random.seed(123)

    # setup job monitoring on Neptune
    ctx = neptune.Context()

    # setup job monitoring on TensorBoardX
    #writer = SummaryWriter()

    # setting device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # setting data set root dir
    root_dir = "/home/SharedFolder/CurrentDatasets/bdd100k"

    # show validation loss
    calc_valid_loss = True

    # data transforms
    transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]), # ImageNet
        "valid": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # ImageNet
    }

    # create data sets
    ds_train = bdd.BDDWeatherDataset(root_dir, split = "bddtrain", transform = transform["train"], force_num = 4)
    ds_valid = bdd.BDDWeatherDataset(root_dir, split = "bddvalid", transform = transform["valid"])

    # data loader
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size = 28, shuffle = True, num_workers = 8)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size = 28, shuffle = True, num_workers = 8)

    # create model
    net = models.resnet50(pretrained = True)
    net.fc = nn.Linear(net.fc.in_features, ds_train._get_num_classes())

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
                eta = (((time.time() - tic) / (epoch + 1)) * (num_epochs - (epoch + 1))) / 3600.0
                if calc_valid_loss:
                    print(f"Epoch {epoch + 1}, Batch {i + 1}: Train Loss = {(running_loss_train / log_step):.3f} "
                          f"Valid Loss = {(running_loss_valid / log_step):.3f} ETA = {eta:.2f}h")
                else:
                    print(f"Epoch {epoch + 1}, Batch {i + 1}: Train Loss = {(running_loss_train / log_step):.3f} "
                          f"ETA = {eta:.2f}h")
                running_loss_train = 0.0
                running_loss_valid = 0.0

            # send to Neptune and TensorBoardX
            ctx.channel_send('train_loss', i, loss.data.cpu().numpy())
            #writer.add_scalar('train/loss', loss.data.cpu().numpy(), i)
            if calc_valid_loss:
                ctx.channel_send('valid_loss', i, loss_valid.data.cpu().numpy())
                # writer.add_scalar('valid/loss', loss_valid.data.cpu().numpy(), i)

    # f1-score on training set, validation set
    f1_train = evaluate_f1_score(net, dl_train)
    f1_valid = evaluate_f1_score(net, dl_valid)
    print(f"F1-Score after {num_epochs} epochs: Training = {f1_train:.2f} Validation = {f1_valid:.2f}")

    # save final model
    torch.save({
        'epochs': num_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': loss.data.cpu().numpy(),
        'train_f1': f1_train,
        'valid_f1': f1_valid
    }, './resnet50_weather_classifier.pth')

    # close TensorBoardX
    #writer.close()
