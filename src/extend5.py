import sys
from bl import *
import pandas as pd 
from extend4 import *
import matplotlib.pyplot as plt
import numpy as np
dataset = sys.argv[1]
selected_raw_data = Data(csv(dataset))

#stts = []
#for _ in range(20):
#    data, test_data = split_data(selected_raw_data, 42*_)
#    model = actLearn(data,shuffle=True)
#    nodes = tree(model.best.rows + model.rest.rows,data)
#    vals = treeFeatureImportance(nodes).values()
#    size = len([i for i in vals if i > 0])
#    #print([round(a,3) for a in vals], size, treeMDI(nodes))
#    stts.append([round(a,3) for a in vals])
#    plt.figure(figsize=(10, 6))
#    plt.bar(range(len(vals)), vals)
#    plt.xlabel('Features')
#    plt.ylabel('Importance')
#    plt.title(f'Feature Importance Distribution (Non-zero features: {size})')
#    plt.grid(True, linestyle='--', alpha=0.7)
#    plt.savefig(f'tmp/bl_{_}.png', dpi=300, bbox_inches='tight')
#    plt.close()
#plt.figure(figsize=(10, 6))
#data_to_plot = [[n[i] for n in stts] for i in range(len(vals))]
#plt.boxplot(data_to_plot, tick_labels=range(len(vals)), showfliers=True)
#plt.xlabel('Features')
#plt.ylabel('Importance')
#plt.title(f'Feature Importance Distribution (Non-zero features: {size})')
#plt.grid(True, linestyle='--', alpha=0.7)
#plt.savefig(f'tmp/bl_total.png', dpi=300, bbox_inches='tight')
#plt.close()

#for ii in range(5):
#    data, test_data = split_data(selected_raw_data, 42*ii)
#    cols = [d.txt for d in data.cols.all]
#    features = [c for c in cols if c[-1] not in ["+", "-", "X"]]
#    shap_FI = run_shap_explainer(data, test_data, features)
#    size = len([shap_FI[i] for i in features if shap_FI[i] > 0])
#    print([round(shap_FI[a],3) for a in features], size)
#    plt.figure(figsize=(10, 6))
#    plt.bar(range(len(features)), [shap_FI[i] for i in features])
#    plt.xlabel('Features')
#    plt.ylabel('Importance')
#    plt.title(f'Feature Importance Distribution (Non-zero features: {size})')
#    plt.grid(True, linestyle='--', alpha=0.7)
#    plt.savefig(f'tmp/shap_{ii}.png', dpi=300, bbox_inches='tight')
#    plt.close()
