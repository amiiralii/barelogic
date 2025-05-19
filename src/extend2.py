from bl import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import matplotlib.pyplot as plt
import random

def split_data(data, test_size=0.2):
    # Shuffle the data
    rows = data.rows.copy()
    random.shuffle(rows)
    
    # Calculate split point
    split_idx = int(len(rows) * (1 - test_size))
    
    # Create train and test Data objects
    train_data = Data([data.cols.names])
    test_data = Data([data.cols.names])
    
    # Add rows to respective datasets
    for row in rows[:split_idx]:
        add(row, train_data)
    for row in rows[split_idx:]:
        add(row, test_data)
        
    return train_data, test_data

def lightgbm(unlabeled, labeled, cols, data, stats, regressor):
    features = [c for c in cols if c[-1] not in ["+","-", "X"]]
    targets = [c for c in cols if c[-1] in ["+","-"]]
    labeled_df = pd.DataFrame(labeled, columns=[c for c in cols])
    unlabeled_df = pd.DataFrame(unlabeled, columns=[c for c in cols])
    le = LabelEncoder()
    for c in cols:
        if c[0].isupper():
            labeled_df[c] = pd.to_numeric(labeled_df[c], errors='coerce')
            unlabeled_df[c] = pd.to_numeric(unlabeled_df[c], errors='coerce')
        else:
            labeled_df[c] = labeled_df[c].astype('category')
            labeled_df[c] = le.fit_transform(labeled_df[c])
            unlabeled_df[c] = unlabeled_df[c].astype('category')
            unlabeled_df[c] = le.fit_transform(unlabeled_df[c])

    params = {
    'objective': 'regression',  # For regression tasks
    'metric': 'mape',           # Root Mean Squared Error
    'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
    'learning_rate': 0.1,       # Learning rate
    'num_leaves': 31,           # Number of leaves in one tree
    'min_data_in_leaf': 1,     # Minimum number of data in a leaf
    'verbose': -1               # Suppress warning messages
    }
    preds = []
    for t in targets:
        if regressor == "lgbm":
            train_data = lgb.Dataset(labeled_df[features], label=labeled_df[t])
            gbm = lgb.train(params, train_data, num_boost_round=100)
            predict = gbm.predict(unlabeled_df[features], num_iteration=gbm.best_iteration)
        elif regressor == "linear":
            model = LinearRegression()
            model.fit(labeled_df[features], labeled_df[t])
            predict = model.predict(unlabeled_df[features])
        preds.append(predict)

    pred_rows = list(zip(*preds))  # shape: (num_unlabeled, num_targets)
    # Create Data object with predictions
    pred_data = Data([targets]+[list(row) for row in pred_rows])
    ydist_values = [ydist(row, pred_data) for row in pred_data.rows]
    top_points = sorted(range(len(ydist_values)), key=lambda i: ydist_values[i])
    top = top_points[:5]
    d2h_results = [ydist(unlabeled[t], data) for t in top]
    return [np.mean(d2h_results), np.std(d2h_results)]

def exp1(file, repeats, regressor = "lgbm"):

    out = {str(j):[] for j in [512, 256,128,64,32,16,8]}
    raw_data  = Data(csv(file))
    data, test_data = split_data(raw_data)
    for Stop in [512, 256,128,64,32,16,8]:
        the.Stop = Stop
        if Stop == 512: the.Stop = len(data.rows)
        stats = []
        for _ in range(repeats):
            model = actLearn(data, shuffle=True)
            labeled = model.best.rows + model.rest.rows
            unlabeled = test_data.rows
            stats.append(lightgbm(unlabeled, labeled, [d.txt for d in data.cols.all], raw_data, stats, regressor))
        mean = sum(s[0] for s in stats) / len(stats)
        std = sum(s[1] for s in stats) / len(stats)
        out[str(Stop)] = [mean,std]
    [print(s,v) for s,v in out.items()]
    return out

def exp2(file, repeats):
    raw_data  = Data(csv(file))
    data, test_data = split_data(raw_data)
    
    out = {str(j):[] for j in [512,256,128,64,32,16,8]}
    for Stop in [512,256,128,64,32,16,8]:
        the.Stop = Stop
        if Stop == 512: the.Stop = len(data.rows)
        stat = []
        for _ in range(repeats):
            model = actLearn(data,shuffle=True)
            nodes = tree(model.best.rows + model.rest.rows,data)
            guesses = sorted([(leaf(nodes,row).ys,row) for row in test_data.rows],key=first)
            d2h_results = [ydist(guess,data) for _,guess in guesses[:5]]
            stat.append([np.mean(d2h_results), np.std(d2h_results)])
        mean = sum(s[0] for s in stat) / len(stat)
        std = sum(s[1] for s in stat) / len(stat)
        out[str(Stop)] = [mean,std]
    [print(s,v) for s,v in out.items()]
    return out
    


dataset = sys.argv[1]
t1 = time.time()
results1 = exp1(file=dataset, repeats=20)
lgbm_time = round(time.time() - t1,3)
print(f'--- {lgbm_time}s ---')
t2 = time.time()
results2 = exp1(file=dataset, repeats=20, regressor="linear")
lr_time = round(time.time() - t2,3)
print(f'--- {lr_time}s ---')
t3 = time.time()
results3 = exp2(file=dataset, repeats=20)
bl_time = round(time.time() - t3,3)
print(f'--- {bl_time}s ---')

# Example: If results are lists of numbers (e.g., ydist values)
x = results1.keys() # or use a meaningful x-axis
plt.errorbar([int(xx) for xx in x], [r[0] for r in results1.values()], yerr=[r[1] for r in results1.values()], label=f'LGBM {lgbm_time} s', marker='o',capsize=5)
plt.errorbar([int(xx)+1 for xx in x], [r[0] for r in results2.values()], yerr=[r[1] for r in results2.values()], label=f'LR {lr_time} s', marker='o',capsize=5)
plt.errorbar([int(xx)+2 for xx in x], [r[0] for r in results3.values()], yerr=[r[1] for r in results3.values()], label=f'BL {bl_time} s', marker='o',capsize=5)

plt.xlabel('Active Learning Budget')
plt.ylabel('Distance to Heaven')
plt.ylim([0.0, 1])
plt.xticks([int(xx) for xx in x], [str(xx) for xx in x])  # Only show the Stop values
plt.title(f'{dataset.split("/")[-1][:-4]} D2H comparison of k Best points')
plt.legend()
plt.savefig(f"result_2/{dataset.split("/")[-1][:-4]}.png", dpi=300)