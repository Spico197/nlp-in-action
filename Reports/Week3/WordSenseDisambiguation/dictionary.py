"""
Lesk(1986): Disambiguation based on the known dictionary.

Suppose a ambiguous word `w` has senses: [s_1, ..., s_k],
and [D_1, ..., D_k] are the definition given by the dictionary.

E_{v_j} is the definition of v_j which is the context word of `w`,
and [s_{j1}, ..., s_{jl}] are the senses of v_j, so E_{v_j} = \cup_{ji}D_{ji}

Coding by Zhu Tong, inspired by *Foundations of Statistical Natural Language Processing*

This is just a **DEMO**.
"""

def overlap(x, y):
    xs = []
    ys = []
    for x_ in x:
        xs.extend(x_)
    for y_ in y:
        ys.extend(y_)

    count = 0
    for xs_ in xs:
        if xs_ in ys:
            count += 1
    return count

w_senses = ['s1', 's2', 's3']
D = [['D_s1_1', 'D_s1_2', 'D_s1_3'],
     ['D_s2_1', 'D_s2_2', 'D_s2_3'],
     ['D_s3_1', 'D_s3_2', 'D_s3_3']]

context = ['c1', 'c2', 'c3']
E_vj= [['D_c1_1', 'D_c1_2', 'D_c1_3'],
     ['D_c2_1', 'D_c2_2', 'D_c2_3'],
     ['D_c3_1', 'D_c3_2', 'D_c3_3']]

scores = []
for sense in w_senses:
    scores.append(overlap(D[w_senses.index(sense)], E_vj))

print("sense: ", w_senses[scores.index(max(scores))])
