from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import numpy as np
import math
from naivebayes import naivebayes
from naivebayes import mean
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def load_data(address):
    dataset = open(address, encoding="utf8").read().split('\n')
    for index, line in enumerate(dataset):
        row_list = line.split(',')
        dataset[index] = [maybe_float(v) for v in row_list]
    data_array = np.array(dataset[:-1], dtype=object)
    return data_array


def maybe_float(element):
    try:
        return float(element)
    except (ValueError, TypeError):
        return element


def split_labels(data_array):
    labels = data_array[:, -1]  # get last column
    data = data_array[:, :-1]  # get all but last column
    return (data, labels)


def preprocessing(data):
    known_values = handle_missing_values(data, '?')
    return known_values


def handle_missing_values(data, mark):
    # mark: missing values mark in dataset
    rows, cols = np.where(data == mark)
    new_data = np.delete(data, rows, 0)
    new_data = np.vstack((new_data,new_data[0])) # append a new row to making suitable length for 6fold cross validation :D
    return new_data


''' 
input: get dataset and k (number of folds) and devide dataset to k fold
output: indices: a list containing end index of each fold 
        folded_data: a list containing tuple of each fold's train and test data '''
def make_kfold(data, k = 3):
    m = len(data)
    indices = []

    if k == 0:
        raise ValueError('k cannot be 0!!')
    if math.fmod(m, k) != 0:
        raise ValueError('Your data cannot be divided by k')
    else:
        fold_len = m/k
    prev_index = fold_len
    for i in range(k):
        indices.append(prev_index)
        prev_index = prev_index + fold_len

    # perform k_fold_cross_validation
    start_index = 0
    dataset = np.asarray(data)
    folded_data = []
    for i in range(0, k):
        end_index = int(indices[i])
        train_data = np.append(dataset[:start_index], dataset[end_index:], axis=0)
        test_data = dataset[start_index:end_index]
        folded_data.append((train_data,test_data))
        start_index = end_index
    return indices, folded_data


def do_cross_validation(folded_data):
    rows, cols = np.shape(folded_data)
    folds_accuracy = []
    folds_proba = []
    predicted_labels = []

    for r in range(rows):
        train = folded_data[r][0]
        test = folded_data[r][1]
        train_data, train_labels = split_labels(train)
        test_data, test_labels = split_labels(test)
        pred_labels, predict_proba = naivebayes(train_data, train_labels, test_data, classes_list)
        accuracy = accuracy_score(test_labels, pred_labels)
        folds_accuracy.append(accuracy)
        folds_proba.append(predict_proba)
        predicted_labels.append(pred_labels)
    return folds_accuracy, folds_proba, predicted_labels


def draw_Roc_diagram(y, scores, pos_label):
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label)
    plt.figure()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Naive Bayes Classifier ROC')
    plt.plot(fpr, tpr, color='blue', lw=2, label='Naive Bayes ROC area ')
    plt.legend(loc="lower right")
    plt.savefig("naive_bayes_roc")
    plt.clf()


if __name__ == '__main__':
    classes_list = ['+','-']
    selected_fold = 5

    dataset = load_data('./dataset/crx.data')
    cleaned_data = preprocessing(dataset)
    (indices, folded_data) = make_kfold(cleaned_data,6) # divide data to 6 fold
    accuracy_list, probability_list, predicted_labels = do_cross_validation(folded_data)
    avg_accuracy = mean(accuracy_list)
    print("accuracy for each fold:", accuracy_list)
    print("average accuracy of folds:", avg_accuracy)
    # ____________ ROC curve for part of data ____________
    test_data, test_labels = split_labels(folded_data[selected_fold][1])
    draw_Roc_diagram(test_labels, probability_list[selected_fold], classes_list[0])
