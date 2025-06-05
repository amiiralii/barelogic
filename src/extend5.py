import sys
from bl import *
import pandas as pd 
from extend4 import split_data
import matplotlib.pyplot as plt

dataset = sys.argv[1]
selected_raw_data = Data(csv(dataset))

for _ in range(20):
    data, test_data = split_data(selected_raw_data)
    model = actLearn(data,shuffle=True)
    nodes = tree(model.best.rows + model.rest.rows,data)
    vals = treeFeatureImportance(nodes).values()
    size = len([i for i in vals if i > 0])
    print([round(a,3) for a in vals], size, treeMDI(nodes))
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(vals)), vals)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Feature Importance Distribution (Non-zero features: {size})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'tmp/feature_importance_{_}.png', dpi=300, bbox_inches='tight')
    plt.close()