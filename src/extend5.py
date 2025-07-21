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
vals = treeFeatureImportance(nodes)
size = len([i for i in vals.values() if i > 0])
print(size)
print(vals)
