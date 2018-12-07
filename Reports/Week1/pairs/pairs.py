#!/usr/bin/python3
#coding=utf8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_pairs(sentences, window=3):
    """找出词对，并计算词对中词的距离、词对出现的均值和方差

    Args:
        sentences <List>: 语料库，iterable对象
        window <int>: 词对窗口

    Returns:
        pairs: <List> 词对
        pairs_count: <dict> 词对词频统计，key为词对的index
        pairs_bias_value: <dict> 每个词对的此位置都为一个列表，如{1: [2, 2, 3], 2: [3, 3, 4]}
        pairs_mean_value: <dict> 词对在窗口条件下的均值
        pairs_variance_value: <dict> 词对在窗口条件下的方差
    
    Raises:
        ValueError("Check pairs"): 当词对的正反序同时存在于pairs中时，抛出此异常
    """
    pairs = []  # 搭配词组
    pairs_count = {}    # 搭配词组的频率统计
    pairs_bias_value = {}
    pairs_mean_value = {}
    pairs_variance_value = {}
    for sentence in sentences:
        sentence = sentence.split()
        for word_index in range(len(sentence)):
            for num in range(1, window + 1, 1):
                if word_index + num <= len(sentence) - 1:
                    pair = [sentence[word_index], sentence[word_index + num]]
                    reverse_pair = [sentence[word_index + num], sentence[word_index]]
                    if pair in pairs and reverse_pair in pairs:
                        print(pair, reverse_pair, pairs)
                        raise ValueError("Check pairs")
                    if reverse_pair in pairs:
                        pair = reverse_pair
                        pairs_bias_value[pairs.index(pair)].append(-num)
                    if pair in pairs:
                        pairs_bias_value[pairs.index(pair)].append(num)                        
                    if pair not in pairs and reverse_pair not in pairs:
                        pairs.append(pair)
                        pairs_bias_value[pairs.index(pair)] = []
                        pairs_bias_value[pairs.index(pair)].append(num)                    
                    pairs_count[pairs.index(pair)] = pairs_count.get(pairs.index(pair), 0) + 1
    for pair_index in range(len(pairs)):
        pairs_mean_value[pair_index] = sum(pairs_bias_value[pair_index])/len(sentences)
        pairs_variance_value[pair_index] = ((np.array(pairs_bias_value[pair_index]) - pairs_mean_value[pair_index])**2).sum()\
                                            /(len(sentences) - 1)
    return pairs, pairs_count, pairs_bias_value, pairs_mean_value, pairs_variance_value


def bar_plot(pair, window, pair_bias_value, pair_mean_value, pair_variance_value):
    """绘制频率随位置变化图

    Args:
        pair: <List> 一个词对，如 ['knocked', 'on']
        window: <int> 词对窗口
        pair_bias_value: <list> 词对中两个词的偏差值
        pair_mean_value: <float> 词对的窗口均值
        pair_variance_value: <float> 词对的窗口方差

    Returns:
        None

    Raises:
        None
    """
    plt.style.use('seaborn')
    plt.rc('font', family='SimHei')

    count = {}
    x_axis = np.linspace(-window, window, window*2 + 1)
    for i in x_axis:
        count[i] = 0
    for val in pair_bias_value:
        count[val] += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x_axis, count.values())
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_axis)
    ax.set_ylabel("频率")
    ax.set_xlabel("{}相对于{}的位置（均值={:.2f}, 方差={:.2f}）".format(pair[1], pair[0], pair_mean_value, pair_variance_value))
    plt.show()


if __name__ == '__main__':
    window = 5 # 二元组搭配查找的窗口
    sentences = [
        "she knocked on his door",
        "she door on his knocked",
        "they knocked at the door",
        "100 women knocked on Donaldson 's door",
        "a man knocked on the metal front door",
    ]

    pairs, pairs_count, pairs_bias_value, pairs_mean_value, pairs_variance_value = get_pairs(sentences, window)
    # print("pairs = \n{}\n\npairs_count = \n{}\n\npairs_bias_value = \n{}\n\npairs_mean_value = \n{}\n\npairs_variance_value = \n{}"\
    #         .format(pairs, pairs_count, pairs_bias_value, pairs_mean_value, pairs_variance_value))
    # print('-'*80)
    # print("pairs = \n{}\n\npairs_count = \n{}\n\npairs_bias_value = \n{}\n\npairs_mean_value = \n{}\n\npairs_variance_value = \n{}"\
    #         .format(['knocked', 'door'], pairs_count[pairs.index(['knocked', 'door'])], 
    #                 pairs_bias_value[pairs.index(['knocked', 'door'])], 
    #                 pairs_mean_value[pairs.index(['knocked', 'door'])], 
    #                 pairs_variance_value[pairs.index(['knocked', 'door'])]))
    pair1 = ['knocked', 'on']
    pair2 = ['she', 'door']
    bar_plot(pair1, window, pairs_bias_value[pairs.index(pair1)], 
                pairs_mean_value[pairs.index(pair1)], 
                pairs_variance_value[pairs.index(pair1)])
    bar_plot(pair2, window, pairs_bias_value[pairs.index(pair2)], 
                pairs_mean_value[pairs.index(pair2)], 
                pairs_variance_value[pairs.index(pair2)])
    print("pairs = \n{}\n\npairs_count = \n{}\n\npairs_bias_value = \n{}\n\npairs_mean_value = \n{}\n\npairs_variance_value = \n{}"\
            .format(pair2, pairs_count[pairs.index(pair2)], 
                    pairs_bias_value[pairs.index(pair2)], 
                    pairs_mean_value[pairs.index(pair2)], 
                    pairs_variance_value[pairs.index(pair2)]))
