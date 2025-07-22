import sys
from bl import *
import pandas as pd 
from extend4 import *
import matplotlib.pyplot as plt
import numpy as np
dataset = sys.argv[1]
raw_data = Data(csv(dataset))
data, test_data = split_data(raw_data)
stp = 32
the.Stop = stp
the.acq = "xploit"
model = actLearn(data,shuffle=True)
nodes = tree(model.best.rows + model.rest.rows,data)
showTree(nodes)

guesses = sorted([(leaf(nodes,row).ys, i) for i, row in enumerate(test_data.rows)],key=first)
print(guesses[0])
print()
used_features = path(nodes, test_data.rows[guesses[0][1]], set())
print(used_features)
#vals = treeFeatureImportance(nodes)
#size = len([i for i in vals.values() if i > 0])
#print(size)
#print(vals)







