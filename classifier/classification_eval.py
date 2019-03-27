import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy import interp
from itertools import cycle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
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


def evaluate_weather(net, data_loader, score_types = ["f1_score_weighted"], cut_off = None, num_batches = None, class_dict = None):
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
        scores = {}
        if "f1_score_weighted" in score_types:
            scores["f1_score_weighted"] = metrics.f1_score(accumulated_targets, accumulated_predictions, average = "weighted")
        if "f1_score_micro" in score_types:
            scores["f1_score_micro"] = metrics.f1_score(accumulated_targets, accumulated_predictions, average = "micro")
        if "f1_score_macro" in score_types:
            scores["f1_score_macro"] = metrics.f1_score(accumulated_targets, accumulated_predictions, average = "macro")
        if "accuracy" in score_types:
            scores["accuracy"] = metrics.accuracy_score(accumulated_targets, accumulated_predictions)
        if "accuracy_balanced" in score_types:
            scores["accuracy_balanced"] = metrics.balanced_accuracy_score(accumulated_targets, accumulated_predictions)
        if "roc_auc_micro" in score_types and class_dict is not None:
            roc_auc, _, _, _ = calculate_roc_auc(accumulated_targets, accumulated_prediction_scores, class_dict)
            scores["roc_auc_micro"] = roc_auc["micro"]
        if "roc_auc_macro" in score_types and class_dict is not None:
            roc_auc, _, _, _ = calculate_roc_auc(accumulated_targets, accumulated_prediction_scores, class_dict)
            scores["roc_auc_macro"] = roc_auc["macro"]
        if "pr_micro" in score_types and class_dict is not None:
            _, _, average_precision, _ = calculate_precision_recall(accumulated_targets, accumulated_prediction_scores, class_dict)
            scores["pr_micro"] = average_precision["micro"]
        if "pr_macro" in score_types and class_dict is not None:
            _, _, average_precision, _ = calculate_precision_recall(accumulated_targets, accumulated_prediction_scores, class_dict)
            scores["pr_macro"] = average_precision["macro"]
        if "mcc" in score_types and class_dict is not None:
            scores["mcc"] = matthews_corrcoef(accumulated_targets, accumulated_predictions)
    net.train()
    return scores, {"targets": accumulated_targets, "predictions": accumulated_predictions, "prediction_scores": accumulated_prediction_scores, "paths": accumulated_paths}


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
    # csoehnel: Remove rows with no true label
    print(cm)
    from itertools import compress
    true_labels = np.sum(cm, axis = 1) > 0
    cm = cm[true_labels, :]
    true_classes = list(compress(classes, true_labels))
    #
    if normalize:
        cm = cm.astype("float") / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)
    fig, ax = plt.subplots(figsize = (10, 10))
    plt.rcParams.update({"font.size": 12})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    im = ax.imshow(cm, interpolation = "nearest", cmap = cmap)
    ax.figure.colorbar(im, ax = ax)
    # We want to show all ticks...
    ax.set(xticks = np.arange(cm.shape[1]),
           yticks = np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels = true_classes,
           title = title,
           ylabel = "True label",
           xlabel = "Predicted label")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")
    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha = "center", va = "center", color = "white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


'''
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
'''
def calculate_roc_auc(y_true, y_scores, class_dict):
    class_ids = sorted([class_dict[class_name] for class_name in class_dict])
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
def plot_roc_curve(roc_auc, fpr, tpr, n_classes, class_dict, title = None):
    class_dict_reverse = {class_id: class_name for class_name, class_id in class_dict.items()}
    # Plot all ROC curves
    lw = 2
    fig, ax = plt.subplots(figsize = (10, 10))
    plt.rcParams.update({"font.size": 12})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    plt.plot(fpr["micro"], tpr["micro"], label = f"micro-average ROC curve (area = {roc_auc['micro']:.2f})", color = 'deeppink', linestyle = ':', linewidth = 4)
    plt.plot(fpr["macro"], tpr["macro"], label = f"macro-average ROC curve (area = {roc_auc['macro']:.2f})", color = 'navy', linestyle = ':', linewidth = 4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color = color, lw = lw, label = f"ROC curve of class: {class_dict_reverse[i]} (area = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw = lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title is not None:
        plt.title(title)
    plt.legend(loc = "lower right")
    return ax


'''
https://github.com/javaidnabi31/Multi-class-with-imbalanced-dataset-classification/blob/master/20-news-group-classification.ipynb
'''
def calculate_precision_recall(y_true, y_scores, class_dict):
    class_ids = sorted([class_dict[class_name] for class_name in class_dict])
    y = np.array(y_true)
    y_test = label_binarize(y, classes = class_ids)
    n_classes = y_test.shape[1]
    y_score = np.array(y_scores)
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score, average = "micro")
    # A "macro-average": quantifying score on each class
    # TBD: Check if calculation correct
    # First aggregate all Recalls
    all_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
    # Then interpolate all PR curves at this points
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        mean_precision += interp(all_recall, recall[i][::-1], precision[i][::-1])
    # Finally average it and compute AUC
    mean_precision /= n_classes
    recall["macro"] = all_recall
    precision["macro"] = mean_precision
    average_precision["macro"] = auc(recall["macro"], precision["macro"])
    return precision, recall, average_precision, n_classes


'''
https://github.com/javaidnabi31/Multi-class-with-imbalanced-dataset-classification/blob/master/20-news-group-classification.ipynb
'''
def plot_pr_curve(precision, recall, average_precision, n_classes, class_dict, title = None):
    class_dict_reverse = {class_id: class_name for class_name, class_id in class_dict.items()}
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    fig, ax = plt.subplots(figsize = (10, 10))
    plt.rcParams.update({"font.size": 12})
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    f_scores = np.linspace(0.2, 0.8, num = 4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color = 'gray', alpha = 0.2)
        plt.annotate(f"f1 = {f_score:.1f}", xy = (0.9, y[45] + 0.02))
    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color = 'gold', lw = 2)
    lines.append(l)
    labels.append(f"micro-average Precision-recall (area = {average_precision['micro']:.2f})")
    l, = plt.plot(recall["macro"], precision["macro"], color = 'brown', lw = 2)
    lines.append(l)
    labels.append(f"macro-average Precision-recall (area = {average_precision['macro']:.2f})")
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color = color, lw = 2)
        lines.append(l)
        labels.append(f"Precision-recall for class: {class_dict_reverse[i]} (area = {average_precision[i]:.2f})")
    fig = plt.gcf()
    fig.subplots_adjust(bottom = 0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if title is not None:
        plt.title(title)
    plt.legend(lines, labels, loc = (.01, .01))
    return ax
