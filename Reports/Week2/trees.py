#coding=utf-8

from math import log
import operator


def load_dataset():
    dataset = [
        [1, 1],
        [1, 1],        
        [1, 0],
        [0, 1],
        [0, 1],
    ]
    labels = [
        'yes',
        'yes',
        'no',
        'no',
        'no',
    ]
    feature_names = [
        'no surfacing',
        'flippers'
    ]
    return dataset, labels, feature_names


class DecisionTreeClassifier(object):
    def __init__(self):
        self.tree = {}
        self.feature_names = []

    def train(self, dataset, labels, feature_names):
        """
        模型训练
        Args:
            dataset: <list> 数据集
            labels: <list> 标签
            feature_names: <list> 特征向量特征值的含义
        Returns:
            tree: <dict> 生成的决策树
        Raises:
            None
        """
        self.feature_names = feature_names  # 这里的feature_names只是方便后续的可视化操作，实际上可以使用自定义feature_names
        self.tree = self._create_tree(dataset, labels, feature_names)
        return self.tree

    def _calculate_shannon_entropy(self, labels):
        """
        计算信息熵

        Args:
            labels: <list> 结果标签
        Returns:
            float: 熵
        Raises:
            None
        """
        number_of_entries = len(labels)
        label_count = {}
        for label in labels:
            label_count[label] = label_count.get(label, 0) + 1
        shannon_entropy = 0.0
        for key in label_count.keys():
            probability = float(label_count[key])/number_of_entries
            shannon_entropy -= probability*log(probability, 2)
        return shannon_entropy


    def _split_dataset(self, dataset, axis, value):
        """
        选取dataset中特征向量的axis位置等于value的向量，且不返回axis所对应的值本身
        即按照value条件抽取数据集中的一部分

        Args:
            dataset: <list> 数据集
            axis: <int> 特征位置
            value: <float/int> 待比对的值
        Returns:
            return_dataset: <list> 抽取后的结果
            return_dataset_index: <list> 抽取后结果所在的序列坐标
        Raises:
            None
        """
        return_dataset = []
        return_dataset_index = []
        for i in range(len(dataset)):
            feature_vector = dataset[i]
            if feature_vector[axis] == value:
                reduced_feature_vector = feature_vector[:axis]
                reduced_feature_vector.extend(feature_vector[axis + 1:])
                return_dataset.append(reduced_feature_vector)
                return_dataset_index.append(i)
        return return_dataset, return_dataset_index


    def _choose_best_feature_to_split(self, dataset, labels):
        """
        根据信息增益选择最佳的决策特征

        Args:
            dataset: <list> 数据集
            labels: <list> 特征值
        Returns:
            best_information_gain: <float> 最佳信息增益
            best_feature: <int> 最佳特征
        Raises:
            None
        """
        feature_length = len(dataset[0])
        base_entropy = self._calculate_shannon_entropy(labels)
        best_information_gain = 0.0
        best_feature = -1
        for i in range(feature_length):
            feature_list = [x[i] for x in dataset]
            uniq_values = set(feature_list)
            new_entropy = 0.0
            for value in uniq_values:
                sub_dataset, sub_dataset_index = self._split_dataset(dataset, i, value)
                probability = len(sub_dataset)/float(len(dataset))
                sub_labels = [labels[i] for i in sub_dataset_index]
                new_entropy += probability*self._calculate_shannon_entropy(sub_labels)
            information_gain = base_entropy - new_entropy
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature = i
        return best_information_gain, best_feature


    def _majority_count(self, labels):
        """
        返回出现次数最多的label

        Args:
            labels: <list> 标签集
        Returns:
            sorted_label_count[0][0]: 拥有最高频率的标签
        Raises:
            None
        """
        label_count = {}
        for vote in labels:
            label_count[vote] = label_count.get(vote, 0) + 1
        sorted_label_count = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_label_count[0][0]


    def _create_tree(self, dataset, labels, feature_names):
        """
        生成决策树

        Args:
            dataset: <list> 数据集
            labels: <list> 标签
            feature_names: <list> 特征向量特征值的含义
        Returns:
            tree: <dict> 生成的决策树
        Raises:
            None
        """
        feature_name = feature_names[:]
        if labels.count(labels[0]) == len(labels):
            return labels[0] # 停止划分的条件：当下面的类别为同一个时
        if len(dataset[0]) == 0:
            return self._majority_count(labels)   # 如果feature的长度为1，即遍历到最后的特征时，返回出现次数最多的类别
        _, best_feature_index = self._choose_best_feature_to_split(dataset, labels)
        best_feature_name = feature_name[best_feature_index]
        tree = {best_feature_name: {}}
        del feature_name[best_feature_index]
        feature_values = [x[best_feature_index] for x in dataset]
        unique_values = set(feature_values)
        for value in unique_values:
            sub_feature_names = feature_name[:]
            sub_dataset, sub_dataset_index = self._split_dataset(dataset, best_feature_index, value)
            sub_labels = [labels[i] for i in sub_dataset_index]
            tree[best_feature_name][value] = self._create_tree(sub_dataset, sub_labels, sub_feature_names)
        return tree


    def classify(self, test_data):
        """
        决策树分类器

        Args:
            test_data: <list> 测试数据的特征向量
        Returns:
            test_label: <str> 测试标签
        Raises:
            None
        """
        test_label = self._predict(self.tree, test_data)
        return test_label

    def _predict(self, tree, test_data):
        """
        决策树分类器

        Args:
            tree: <dict> 树
            test_data: <list> 测试数据的特征向量
        Returns:
            test_label: <str> 测试标签
        Raises:
            None
        """
        test_label = ""
        first_key = list(tree.keys())[0] # dict的keys方法取得的是dict_keys对象，所以list不支持直接进行后续的index
        inner_dict = tree[first_key]
        feature_index = self.feature_names.index(first_key)
        for key in inner_dict.keys():
            if test_data[feature_index] == key:
                if type(test_data).__name__ == 'dict':
                    test_label = self._predict(inner_dict[key], test_data)
                else:
                    test_label = inner_dict[key]
        return test_label


if __name__ == '__main__':
    dataset, labels, feature_names = load_dataset()
    # shannon_entropy = calculate_shannon_entropy(labels)
    # gain, feature = choose_best_feature_to_split(dataset, labels)
    # print(gain, feature)
    clf = DecisionTreeClassifier()
    clf.train(dataset, labels, feature_names)
    result = clf.classify([2, 1])
    print(result)
