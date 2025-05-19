from bl import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import matplotlib.pyplot as plt

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
            lr = LinearRegression()
            lr.fit(labeled_df[features], labeled_df[t])
            predict = lr.predict(unlabeled_df[features])
        preds.append(predict)
    b4    = yNums(data.rows, data) 
    pred_rows = list(zip(*preds))  # shape: (num_unlabeled, num_targets)
    # Create Data object with predictions
    pred_data = Data([targets]+[list(row) for row in pred_rows])
    ydist_values = [ydist(row, pred_data) for row in pred_data.rows]
    top_points = sorted(range(len(ydist_values)), key=lambda i: ydist_values[i])
    for k in [20,15,10,5,3,1]:
        top = top_points[:k]
        d2h_results = [ydist(unlabeled[t], data) for t in top]
        stats[k] += [np.mean(d2h_results)]

    return stats

def exp1(file, repeats, regressor = "lgbm"):
    raw_data  = Data(csv(file))
    data, test_data = split_data(raw_data)
    b4    = yNums(data.rows, raw_data) 
    overall= {j:Num() for j in [256,128,64,32,16,8]}
    overall_results = {j:[] for j in [256,128,64,32,16,8]}
    for Stop in overall:
        the.Stop = Stop
        stats = {j:[] for j in [20,15,10,5,3,1]}
        for _ in range(repeats):
            model = actLearn(data, shuffle=True)
            labeled = model.best.rows + model.rest.rows
            unlabeled = test_data.rows
            stats = lightgbm(unlabeled, labeled, [d.txt for d in data.cols.all], raw_data, stats, regressor)
        
        for s in stats:
            mean = sum(stats[s]) / len(stats[s])
            stats[s] = mean
        def win(x): return round(100*(1 - (x - b4.lo)/(b4.mu - b4.lo)))
        overall_results[Stop].append([win(stats[k]) for k in stats])
    return overall_results

def exp2(file,repeats=30, smart=True):
    raw_data  = Data(csv(file))
    data, test_data = split_data(raw_data)
    b4    = yNums(data.rows, raw_data) 
    overall= {j:Num() for j in [256,128,64,32,16,8]}
    overall_results = {j:[] for j in [256,128,64,32,16,8]}
    for Stop in overall:
        the.Stop = Stop
        after = {j:Num() for j in [20,15,10,5,3,1]}
        learnt = Num()
        rand =Num()
        for _ in range(repeats):
            model = actLearn(data,shuffle=True)
            nodes = tree(model.best.rows + model.rest.rows,data)
            add(ydist(model.best.rows[0],data), learnt)
            guesses = sorted([(leaf(nodes,row).ys,row) for row in test_data.rows],key=first)
            for k in after:
                if smart:
                    smart = min([(ydist(guess,data),guess) for _,guess in guesses[:k]], 
                                key=first)[1]
                    selection = sum( ydist(guess,data) for _,guess in guesses[:k]) / k
                    add(selection,after[k]) 
                else:
                    dumb = min([(ydist(row,data),row) for row in random.choices(test_data.rows,k=k)],
                        key=first)[1]
                    add(ydist(dumb,data),after[k]) 
        def win(x): return round(100*(1 - (x - b4.lo)/(b4.mu - b4.lo)))
        overall_results[Stop].append([win(after[k].mu) for k in after])
    return overall_results

def make_chart(results, time, title):
    # Example: If results are lists of numbers (e.g., ydist values)
    x = results.keys() # or use a meaningful x-axis
    # Plot 1: LGBM results
    plt.figure(figsize=(10, 6))
    for i, k in enumerate([20, 15, 10, 5, 3, 1]):
        plt.plot(x, [results[r][0][i] for r in results], label=f'k={k}', marker='o')
    plt.xlabel('Active Learning Budget')
    plt.ylabel('win ("%"optimal ) ')
    plt.ylim([-20, 100])
    plt.xticks([int(xx) for xx in x], [str(xx) for xx in x])
    plt.title(f'{title} Results (Time: {time}s)')
    plt.legend()
    plt.savefig(f"result_3/{dataset.split('/')[-1][:-4]}_{title}.png", dpi=300)
    plt.close()

def save_all_results_to_csv(results1, results2, results3, filename):
    # Create a DataFrame with all results
    data = []
    
    # First add all LGBM results
    for stop in sorted(results1.keys()):
        for k_values in results1[stop]:
            row = {'Stop': stop, 'Method': 'LGBM'}
            for i, k in enumerate([20, 15, 10, 5, 3, 1]):
                row[f'k={k}'] = k_values[i]
            data.append(row)
    
    # Then add all Linear Regression results
    for stop in sorted(results2.keys()):
        for k_values in results2[stop]:
            row = {'Stop': stop, 'Method': 'LR'}
            for i, k in enumerate([20, 15, 10, 5, 3, 1]):
                row[f'k={k}'] = k_values[i]
            data.append(row)
    
    # Finally add all BareLogic results
    for stop in sorted(results3.keys()):
        for k_values in results3[stop]:
            row = {'Stop': stop, 'Method': 'BL'}
            for i, k in enumerate([20, 15, 10, 5, 3, 1]):
                row[f'k={k}'] = k_values[i]
            data.append(row)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(f"result_3/{filename}.csv", index=False)

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

make_chart(results1, lgbm_time, "LGBM")
make_chart(results2, lr_time, "LR")
make_chart(results3, bl_time, "BL")

# After running the experiments, save all results to a single CSV
dataset_name = dataset.split('/')[-1][:-4]
save_all_results_to_csv(results1, results2, results3, f"{dataset_name}")
