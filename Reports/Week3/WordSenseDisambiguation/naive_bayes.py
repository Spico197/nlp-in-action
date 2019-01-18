"""
Word Sense Disambiguation with Naive Bayes Classifier

Use a certain word: `drug` as the ambiguous word.
From the dataset, we can get:
    - `drug` has 4 meanings:
        - AC:02:e         Health and diseases -> Medicines/physic
        - BJ:01:l         Trade and finance -> Selling
        - BH:12:b:02      Travel and travelling -> Means of travel -> Vehicle
        - AI:10:a         Physical sensation -> Use of drugs, poison -> Drugging a person/thing
The indicator words:
    - `medication`: prices, prescription, patent, increase, consumer, pharmaceutical

Dataset:
    - [hansard](https://www.hansard-corpus.org)

In this demo, we will use `drug` and its indicator words on `medication`
to train a classifier by `Naive Bayes Algorithm`.

Coding by Zhu Tong, inspired by *Foundations of Statistical Natural Language Processing*
"""

import os
from math import log

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer


def get_dataset(filename):
    # load dataset from `filename`
    dataset = pd.read_excel(filename, index_col=0)
    return dataset

def data_preprocessing(dataset, ambiguous_word='drug', context_window=3):
    """
    Get the context tokens which are cut off by a certain window.
    Get the contents after word-tokenized, stemming and punctuations-removed
    :param dataset: data set, pandas.DataFrame format
    :param context_window: the word number before(after) the ambiguous word
    :return: contents and context words
    """
    contents = []
    context_words = []

    punctuations = set('`~!@#$%^&*()_+-=[]\{}|;:",./<>?')

    st = LancasterStemmer() # stemmer

    # print(dataset.columns)
    for ind in dataset.index:
        row = dataset.loc[ind, ['content', 'class']]
        contents.append(
            list(
                map(
                    st.stem, # token stemming
                    filter( # remove punctuations
                        lambda w: True if w not in punctuations else False,
                          word_tokenize(row['content'])
                    )
                )
            )
        )

    for content in contents:
        contexts = []
        try:
            ind = content.index(ambiguous_word)
            # boundary conditions
            if ind + context_window > len(content) - 1:
                contexts.extend(content[ind+1:])
            else:
                contexts.extend(content[ind+1: ind+context_window+1])
            if ind - context_window < 0:
                contexts.extend(content[0:ind])
            else:
                contexts.extend(content[ind-context_window: ind])
        except ValueError:
            pass    # if there is no such ambiguous word in content, pass
        finally:
            context_words.append(contexts)

    return contents, context_words

def cal_count_v_sk(contexts, indicators):
    count_v_sk = []

    for indicator in indicators:
        count = 0
        for context in contexts:
            if indicator in context:
                count += 1
        count_v_sk.append(count)

    return count_v_sk

def cal_count_sk(sense, senses):
    count = 0
    for s in senses:
        if sense == s:
            count += 1
    return count

def cal_count_w(ambiguous_word, contents):
    count = 0

    for content in contents:
        if ambiguous_word in content:
            count += 1

    return count

def get_contexts_by_sense(sense, senses, contexts):
    context = []

    for s, ct in zip(senses, contexts):
        if sense == s:
            context.append(ct)

    return context


class NaiveBayesDisambiguater(object):
    def __init__(self, ambiguous_word, senses, indicators, contents, contexts):
        self.ambiguous_word = ambiguous_word
        self.senses = senses    # NOTICE: the senses is dataset['sense'], which is a whole column data
        self.indicators = indicators
        self.contents = contents
        self.contexts = contexts

        self.probs_sk = []
        self.probs_v_sk = []  # two dimensional list


    def cal_count_v_sk(self, contexts):
        count_v_sk = []

        for indicator in self.indicators:
            count = 0
            for context in contexts:
                if indicator in context:
                    count += 1
            count_v_sk.append(count)

        return count_v_sk

    def cal_count_sk(self, sense):
        count = 0
        for s in self.senses:
            if sense == s:
                count += 1
        return count

    def cal_count_w(self):
        count = 0

        for content in self.contents:
            if self.ambiguous_word in content:
                count += 1

        return count

    def get_contexts_by_sense(self, sense, contexts=list()):
        context = []
        if not contexts:
            contexts = self.contexts
        for s, ct in zip(self.senses, contexts):
            if sense == s:
                context.append(ct)

        return context

    def train(self):
        self.probs_sk = []
        self.probs_v_sk = []  # two dimensional list

        senses_set = list(set(senses))
        for sense in senses_set:
            self.probs_sk.append(self.cal_count_sk(sense) \
                                 / self.cal_count_w())
            probs_v_sk.append(self.cal_count_v_sk(self.get_contexts_by_sense(sense)))

    def test(self, context_words):
        probs_v_sk = []  # two dimensional list

        senses_set = list(set(self.senses))
        for sense in senses_set:
            probs_v_sk.append(self.cal_count_v_sk(self.get_contexts_by_sense(sense, context_words)))

        # print("test: P(v | s_k): ", probs_v_sk)
        scores = []
        for p_sk, p_v_sk in zip(self.probs_sk, probs_v_sk):
            score = log(p_sk)
            for p_v in p_v_sk:
                if p_v == 0:
                    p_v = 1e-12
                score += log(p_v)
            scores.append(score)

        max_score = max(scores)
        return senses_set[scores.index(max_score)]


if __name__ == '__main__':
    ambiguous_word = 'drug'
    context_window = 3
    indicator_words = ['pric', 'prescrib', 'pat', 'increas', 'consum', 'pharmaceut'] # after stemming

    filename = 'drug_word_disambiguation.xlsx'
    data = get_dataset(os.path.join(os.path.dirname(__file__), filename))
    contents, context_words = data_preprocessing(data, ambiguous_word=ambiguous_word, context_window=3)

    """Training"""
    probs_sk = []
    probs_v_sk = [] # two dimensional list

    senses = data['class']
    senses_set = list(set(senses))
    for sense in senses_set:
        probs_sk.append(cal_count_sk(sense, senses)/cal_count_w(ambiguous_word, contents))
        probs_v_sk.append(cal_count_v_sk(get_contexts_by_sense(sense, senses, context_words), indicator_words))

    print("P(s_k): ", probs_sk)
    print("P(v | s_k): ", probs_v_sk)

    """Disambiguation"""
    # # test on train data - nonsense
    # scores = []
    # for p_sk, p_v_sk in zip(probs_sk, probs_v_sk):
    #     score = log(p_sk)
    #     for p_v in p_v_sk:
    #         if p_v == 0:
    #             p_v = 1e-12
    #         score += log(p_v)
    #     scores.append(score)
    #
    # max_score = max(scores)
    # print("sense: ", senses_set[scores.index(max_score)])

    # test set 1
    context_words = ['a', 'b', 'c', 'd', 'e', 'f']
    probs_v_sk = []  # two dimensional list

    senses = data['class']
    senses_set = list(set(senses))
    for sense in senses_set:
        probs_v_sk.append(cal_count_v_sk(get_contexts_by_sense(sense, senses, context_words), indicator_words))

    print("test: P(v | s_k): ", probs_v_sk)
    scores = []
    for p_sk, p_v_sk in zip(probs_sk, probs_v_sk):
        score = log(p_sk)
        for p_v in p_v_sk:
            if p_v == 0:
                p_v = 1e-12
            score += log(p_v)
        scores.append(score)

    max_score = max(scores)
    print("sense: ", senses_set[scores.index(max_score)])

    # test set 2
    context_words = ['pric', 'prescrib', 'pat', 'increas', 'consum', 'pharmaceut']
    probs_v_sk = []  # two dimensional list

    senses = data['class']
    senses_set = list(set(senses))
    for sense in senses_set:
        probs_v_sk.append(cal_count_v_sk(get_contexts_by_sense(sense, senses, context_words), indicator_words))

    print("test: P(v | s_k): ", probs_v_sk)
    scores = []
    for p_sk, p_v_sk in zip(probs_sk, probs_v_sk):
        score = log(p_sk)
        for p_v in p_v_sk:
            if p_v == 0:
                p_v = 1e-12
            score += log(p_v)
        scores.append(score)

    max_score = max(scores)
    print("sense: ", senses_set[scores.index(max_score)])

    nbd = NaiveBayesDisambiguater(ambiguous_word, data['class'], indicator_words, contents, context_words)
    nbd.train()
    print("test by NaiveBayesDisambiguater class: ",
          nbd.test(['pric', 'prescrib', 'pat', 'increas', 'consum', 'pharmaceut']))


""" Result
P(s_k):  [0.30959752321981426, 0.07120743034055728, 0.30959752321981426, 0.30959752321981426]
P(v | s_k):  [[1, 1, 0, 2, 1, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0], [0, 1, 0, 0, 0, 0]]
test: P(v | s_k):  [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
sense:  AC:02:e
test: P(v | s_k):  [[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
sense:  AC:02:e
test by NaiveBayesDisambiguater class:  AC:02:e
"""
