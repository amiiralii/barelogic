import sys
from bl import *
import pandas as pd 
from extend4 import *
import matplotlib.pyplot as plt
import numpy as np
dataset = sys.argv[1]
selected_raw_data = Data(csv(dataset))

data, test_data = split_data(selected_raw_data, 95)
model = actLearn(data,shuffle=True)
nodes = tree(model.best.rows + model.rest.rows,data)
showTree(nodes)
vals = treeFeatureImportance(nodes)
size = len([i for i in vals.values() if i > 0])
print(size)
print(vals)
