#coding=utf-8
import re
import operator


def load_data(data_path, lines=5, ngram=3):
    unseen_value = "UNSEEN_VALUE"
    combination_count = {}
    combinations = []
    with open(data_path, 'r', encoding='utf-8') as file:
        count = 0
        for line in file:
            line = line.lower()
            if count < lines:
                line_without_punctuations = [word for word in line.split() if not set([word]) & set('~!@#$%^&*()_+{}:"|<>?`[];\'\,./\\')]
                # padding
                for i in range(ngram - 1): # 若n-gram为3，则前后各补两个tokens
                    line_without_punctuations.append(unseen_value)
                    line_without_punctuations.insert(0, unseen_value)
                # combination
                for word_start_index in range(0, len(line_without_punctuations) - ngram + 1):
                    combination = []
                    for word_index in range(word_start_index, word_start_index + ngram):
                        combination.append(line_without_punctuations[word_index])
                    combinations.append(combination)
            else:
                break
            count += 1
            print("Process: {}/{}".format(count, lines))
    for combination in combinations:
        combination_count[str(combination)] = combination_count.get(str(combination), 0) + 1
    return combinations, combination_count


def get_probability(prefix_words, combinations, combination_count):
    words = prefix_words[:]
    str_words = ", ".join(["'{}'".format(word) for word in words])
    words_count = {}
    probabilities = {}
    count = 0
    for key, item in combination_count.items():
        extract_string = key[1:len(str_words) + 1]
        if extract_string == str_words:
            # print(extract_string, str_words, key)
            word = re.match(r"\[.*, ['\"](.*?)['\"]\]", key).group(1)
            words_count[word] = words_count.get(word, 0) + item
            # print(words_count)
    for key, item in words_count.items():
        probabilities[key] = item/len(combinations)
    sorted_probabilities = sorted(probabilities.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_probabilities


if __name__ == '__main__':
    bllip_path = 'D:\\NLP\\bllip\\bllip\\bllip.sents'
    # combinations, combination_count = load_data('data/data.txt', lines=500000, ngram=3)
    combinations, combination_count = load_data(bllip_path, lines=500000, ngram=3)
    print(combination_count)
    probability = get_probability(['it', 'is'], combinations, combination_count)
    print(probability)
    """ 结果示例
    ("n't", 6.853739500487236e-05), 
    ('a', 3.541523385010133e-05), 
    ('the', 2.836615852741929e-05), 
    ('not', 1.825965294429685e-05), 
    """
