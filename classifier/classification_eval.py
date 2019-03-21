import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy import interp
from itertools import cycle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

''' depricated '''
def evaluate_timeofday(net, data_loader, num_batches = None):
    net.eval() # disables dropout, etc.
    with torch.no_grad(): # temporarily disables gradient computation for speed-up
        accumulated_targets = []
        accumulated_outputs = []
        for i, data in enumerate(data_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            accumulated_targets.extend(targets.detach().cpu().numpy().tolist())
            # f1 score: https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel
            accumulated_outputs.extend((outputs.detach().cpu().numpy() > 0.5).tolist())
            if num_batches is not None and (i >= (num_batches - 1)):
                break
        f1_score = metrics.f1_score(accumulated_targets, accumulated_outputs, average="weighted")
    net.train()
    return f1_score


def evaluate_weather(net, data_loader, score_type = "f1_score_weighted", cut_off = None, num_batches = None):
    net.eval() # disables dropout, etc.
    with torch.no_grad(): # temporarily disables gradient computation for speed-up
        accumulated_targets = []
        accumulated_prediction_scores = []
        accumulated_paths = []
        for i, data in enumerate(data_loader, 0):
            inputs, targets, paths = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            accumulated_targets.extend(targets.detach().cpu().numpy().tolist())
            predictions = torch.exp(nn.LogSoftmax(dim = 1)(outputs)).detach().cpu().numpy()
            accumulated_prediction_scores.extend(predictions.tolist())
            accumulated_paths.extend(paths)
            if num_batches is not None and (i >= (num_batches - 1)):
                break
        if cut_off is not None:
            accumulated_predictions = [np.argmax(sample_scores) if max(sample_scores) >= cut_off else -1 for sample_scores in accumulated_prediction_scores]
        else:
            accumulated_predictions = np.argmax(accumulated_prediction_scores, axis = 1)
        # f1 score: https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel
        # weighted = macro average with class balancing
        if score_type == "f1_score_weighted":
            score = metrics.f1_score(accumulated_targets, accumulated_predictions, average = "weighted")
        elif score_type == "f1_score_micro":
            score = metrics.f1_score(accumulated_targets, accumulated_predictions, average = "micro")
        elif score_type == "f1_score_macro":
            score = metrics.f1_score(accumulated_targets, accumulated_predictions, average = "macro")
        elif score_type == "accuracy":
            score = metrics.accuracy_score(accumulated_targets, accumulated_predictions)
        else:
            raise Exception("Unknown score_type")
    net.train()
    return score, {"targets": accumulated_targets, "predictions": accumulated_predictions, "prediction_scores": accumulated_prediction_scores, "paths": accumulated_paths}


'''
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
'''
def plot_confusion_matrix(y_true, y_pred, classes, normalize = False, title = None, cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation = 'nearest', cmap = cmap)
    ax.figure.colorbar(im, ax = ax)
    # We want to show all ticks...
    ax.set(xticks = np.arange(cm.shape[1]),
           yticks = np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels = classes, yticklabels = classes,
           title = title,
           ylabel = 'True label',
           xlabel = 'Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha = "center", va = "center", color = "white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


'''
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
'''
def calculate_roc_auc(class_ids, y_true, y_scores):
    y = np.array(y_true)
    y_test = label_binarize(y, classes = class_ids)
    n_classes = y_test.shape[1]
    y_score = np.array(y_scores)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return roc_auc, fpr, tpr, n_classes


'''
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
'''
def plot_roc_curve(roc_auc, fpr, tpr, n_classes):
    # Plot all ROC curves
    lw = 2
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label = 'micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color = 'deeppink', linestyle = ':', linewidth = 4)
    plt.plot(fpr["macro"], tpr["macro"], label = 'macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color = 'navy', linestyle = ':', linewidth = 4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color = color, lw = lw, label = 'ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw = lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc = "lower right")
    plt.show()
