import numpy as np
import collections
from math import sqrt
from math import pi
from math import exp

float_precision = '.6f'


def naivebayes(train_data, train_labels, test_data, classes):
    # prior_prob_list: list of each class probability
    # posterior_prob_list: list of each class posterior probability for each test sample
    pred_labels = []
    scores = []
    prior_prob_list = calculate_prior_probability(train_labels, classes)
    for test_sample in test_data:
        posterior_prob_list = calculate_posterior_probability(train_data, train_labels, test_sample, classes)
        class_score = np.multiply(prior_prob_list, posterior_prob_list)
        scores.append(class_score)
        pred_label = classes[np.argmax(class_score)]
        pred_labels.append(pred_label)
    scores = np.array(scores)
    pred_proba = predict_proba(scores, 0)  # 0: index of positive class
    return pred_labels, pred_proba


# calculate prior probability for each class
def calculate_prior_probability(labels, classes):
    prob_list = []
    total = len(labels)
    for cls in classes:
        count = np.count_nonzero(labels == cls)
        prob_list.append(count/total)
    return prob_list


# calculate posterior probability for each class
def calculate_posterior_probability(features, labels, test_sample, classes):
    posterior_prob_list = []
    rows, cols = np.shape(features)

    for cls in classes:
        features_prob_list = []
        for feature_index in range(cols):
            feature_values = features[:,feature_index]

            # calculate probability for discrete features
            if type(feature_values[0]) is str:
                prob = calc_discrete_feature_prob(feature_values, labels, test_sample[feature_index], cls)
                features_prob_list.append(prob)
            # calculate probability for continuous  features
            elif type(feature_values[0]) is float or type(feature_values[0]) is int:
                prob = calc_continuous_feature_prob(feature_values, labels, test_sample[feature_index], cls)
                features_prob_list.append(prob)
            else:
                raise TypeError('these type of features has not been handled!')
        probs_product = np.prod(features_prob_list)
        posterior_prob_list.append(probs_product)
    return posterior_prob_list


def calc_discrete_feature_prob(feature_vector, labels, test_sample, cls):
    global float_precision
    sample_count = 0
    k = 1 # laplace smoothing strength
    cls_count = np.count_nonzero(labels == cls)
    sample_possible_count = collections.Counter(feature_vector) # for laplace smoothing

    for index, value in enumerate(feature_vector):
        if labels[index] == cls and value == test_sample:
            sample_count = sample_count + 1
    # laplace smoothing:
    prob = sample_count + k/(cls_count + len(sample_possible_count)*k)
    return prob


def calc_continuous_feature_prob(feature_vector, labels, test_sample, cls):
    global float_precision
    values = []

    for index, f in enumerate(feature_vector):
        if labels[index] == cls:
            values.append(f)
    avg = mean(values)
    variance = stdev(values)
    prob = gaussian_prob_distribution(test_sample, avg, variance)
    return prob


# Calculate the mean of a list of numbers
def mean(numbers):
    avg = sum(numbers)/float(len(numbers))
    return avg


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers))
    return sqrt(variance)


# Calculate the Gaussian probability distribution function for x
def gaussian_prob_distribution(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    if exponent == 0:
        exponent = np.finfo(float).tiny
    prob = (1 / (sqrt(2 * pi) * stdev)) * exponent
    return prob


def predict_proba(scores, positive_class_index):
    positive_class_prob = scores[:,positive_class_index]
    total = np.sum(scores, axis=1)
    probability = np.divide(positive_class_prob,total)
    return probability
