#!/usr/bin/python3
#coding=utf8

import numpy as np
import operator


def create_dataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataset, labels, k):
    dataset_size = dataset.shape[0]
    distances = ((np.tile(inX, (dataset_size, 1)) - dataset)**2).sum(axis=1)**0.5
    sorted_distance_indices = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distance_indices[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # sorted_class_count = [('B', 2), ('A', 1)]
    return sorted_class_count[0][0]

def main():
    group, labels = create_dataset()
    result = classify0([0, 0], group, labels, 3)
    print(result)

if __name__ == '__main__':
    main()