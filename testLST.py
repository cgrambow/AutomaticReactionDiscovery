from node import Node
from interpolation import LST

number = [6, 1, 1, 1, 1]
coords_start = [[-1.27136748,   -0.06410256,    0.00000000],
                [-0.91471305,   -1.07291257,    0.00000000],
                [-0.91469464,    0.44029563,    0.87365150],
                [-0.91469464,    0.44029563,   -0.87365150],
                [-2.34136748,   -0.06408938,    0.00000000]]
node_start = Node(coords_start, number)

coords_end = [[-1.44970390,   -0.31630166,   -0.43682575],
              [-1.09304947,   -1.32511166,   -0.43682575],
              [-0.73635822,    0.69249472,    1.31047726],
              [-1.09303106,    0.18809653,   -1.31047726],
              [-2.51970390,   -0.31628847,   -0.43682575]]
node_end = Node(coords_end, number)

LSTtest = LST(node_start, node_end)

LSTpath = LSTtest.getLSTpath()
for node in LSTpath:
    print node
    print '\n'
