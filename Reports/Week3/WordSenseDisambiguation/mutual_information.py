"""
Brown et al. 1991: `Flip-Flop Algorithm` to make mutual information increase monotonically

Suppose we are going to translate `prendre` based on its object,
we have `t = [t_1, ..., t_m] = ['take', 'make', 'rise', 'speak']`
and `x = [x_1, ..., x_n] = ['mesure', 'note', 'example', 'décision', 'parole']`

Coding by Zhu Tong, inspired by *Foundations of Statistical Natural Language Processing*

Reference Materials:
    - [NLP第十篇-语义分析](https://www.jianshu.com/p/7463267b0106)
"""


from itertools import combinations
from random import random
from math import log


def get_partition(number, group):
    for item in combinations(group, number):
        yield item

def cal_numtual_information(x, y):
    """Pseudo Mutual Information Values"""
    P_x = [random() for _ in x]
    P_y = [random() for _ in y]
    xy = []
    for x_ in x:
        for y_ in y:
            xy.append([x_, y_])
    P_xy = [random() for _ in xy]

    I_mul = 0.0
    for x_ in x:
        for y_ in y:
            I_mul += P_xy[xy.index([x_, y_])] * \
                     log(P_xy[xy.index([x_, y_])] \
                         / (P_x[x.index(x_)]*P_y[y.index(y_)]))
    return I_mul


if __name__ == '__main__':
    t = ['take', 'make', 'rise', 'speak']
    x = ['mesure', 'note', 'example', 'décision', 'parole']

    p = [t[0], t[1]]
    q = []
    p_max = p[:]
    q_max = []
    ps = list(get_partition(2, t))
    qs = list(get_partition(2, x))
    I = 0.0
    I_new = 1e-12
    while I_new - I > 0:
        I_t = [cal_numtual_information(p_max, q) for q in qs]
        q_max = qs[I_t.index(max(I_t))]
        I_t = [cal_numtual_information(p, q_max) for p in ps]
        p_max = ps[I_t.index(max(I_t))]
        I_new = cal_numtual_information(q_max, p_max)

    print("q_max: ", q_max)
    print("p_max: ", p_max)
