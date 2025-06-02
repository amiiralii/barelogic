from bl import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR as SupportVectorRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import random
import lime
import lime.lime_tabular
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
from IPython.display import display, HTML
from neural_net import *
from relieff import RReliefF


Random_seed = 42

def split_data(data, test_size=0.2):
    global Random_seed
    # Shuffle the data
    rows = data.rows.copy()
    random.seed(Random_seed)
    Random_seed *= 2
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

def run_bl_explainer(data, test_data, features):
    """Run BareLogic explainer and return feature importance"""
    the.Stop = 32
    t1 = time.time()
    model = actLearn(data, shuffle=True)
    nodes = tree(model.best.rows + model.rest.rows, data)
    #showTree(nodes)
    print("MDI of BL tree:\t", round(treeMDI(nodes), 3))
    bl_FI = treeFeatureImportance(nodes)
    w = sum(vf for vf in bl_FI.values())
    for f in bl_FI:
        bl_FI[f] = bl_FI[f] / w
    bl_FI["explainer"] = "BL"
    
    return bl_FI, time.time()-t1

def distribution_plot(labeled, data, features, sample):
    cols = [d.txt for d in data.cols.all]
    
    labeled_df = pd.DataFrame(labeled, columns=cols)
    labeled_y = pd.DataFrame([ydist(row, data) for row in labeled], columns=["d2h"])
    
    plt.figure(figsize=(15, 10))
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(labeled_df[feature], labeled_y['d2h'], alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel('d2h')
        plt.title(f'{feature} vs d2h')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'explanations/{sys.argv[1].split("/")[-1][:-4]}/feature_scatter_plot_{sample}.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_lime_explainer(data, test_data, features, idx=5):
    """Run LIME explainer and return feature importance"""
    print('-------LIME:-------')
    
    # Prepare data
    labeled = data.rows
    unlabeled = test_data.rows
    cols = [d.txt for d in data.cols.all]
    
    labeled_df = pd.DataFrame(labeled, columns=cols)
    unlabeled_df = pd.DataFrame(unlabeled, columns=cols)
    labeled_y = pd.DataFrame([ydist(row, data) for row in labeled], columns=["d2h"])
    unlabeled_y = pd.DataFrame([ydist(row, data) for row in unlabeled], columns=["d2h"])
    
    
    # Preprocess data
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
    
    # Train LGBM model
    model = LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        num_leaves=8,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=50,
        min_data_in_leaf=2,
        min_data_in_bin=1,
        max_bin=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        force_col_wise=True,
        verbose=-1,
        random_state=42
    )
    model.fit(labeled_df[features], labeled_y)
    
    # LIME explanation
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=labeled_df[features].values,
        feature_names=features,
        categorical_features=[c for c in features if not c[0].isupper()],
        mode='regression',
        discretize_continuous=True
    )
    
    instance = unlabeled_df[features].iloc[idx].values.reshape(1, -1)
    true_value = unlabeled_y.iloc[idx]
    predicted_value = model.predict(instance)[0]
    
    print(f"True target = {true_value['d2h']}, Model prediction = {predicted_value}")
    
    exp = explainer.explain_instance(
        data_row=instance.flatten(),
        predict_fn=model.predict,
        num_features=6
    )
    
    print("LIME Explain:")
    for feature, weight in exp.as_list():
        direction = "↑" if weight > 0 else "↓"
        print(f"{feature:>20}: {weight:+.3f} ({direction})")
    
    exp.save_to_file(f'explanations/{sys.argv[1].split("/")[-1][:-4]}/lgbm_lime_explanation.html')
    
    # Calculate feature importance
    total_gain = model.booster_.feature_importance(importance_type='gain')
    n_trees = model.booster_.num_trees()
    mdi = total_gain / n_trees
    
    mdi_df = pd.DataFrame({
        'feature': features,
        'total_gain': total_gain,
        'mdi': mdi
    }).sort_values('mdi', ascending=False)
    
    mdi_df['mdi_norm'] = mdi_df['mdi'] / mdi_df['mdi'].sum()
    print(mdi_df[['feature', 'mdi_norm']])
    
    lime_FI = {"explainer": "lime"}
    for index, row in mdi_df.iterrows():
        lime_FI[row['feature']] = row['mdi_norm']
    
    return lime_FI

def run_shap_explainer(data, test_data, features, idx=5):
    """Run SHAP explainer and return feature importance"""
    print('-------SHAP:-------')
    t1=time.time()
    # Prepare data
    labeled = data.rows
    unlabeled = test_data.rows
    cols = [d.txt for d in data.cols.all]
    
    labeled_df = pd.DataFrame(labeled, columns=cols)
    unlabeled_df = pd.DataFrame(unlabeled, columns=cols)
    
    # Train LGBM model
    model = LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        num_leaves=8,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=50,
        min_data_in_leaf=2,
        min_data_in_bin=1,
        max_bin=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        force_col_wise=True,
        verbose=-1,
        random_state=42
    )
    
    labeled_y = pd.DataFrame([ydist(row, data) for row in labeled], columns=["d2h"])
    model.fit(labeled_df[features], labeled_y)
    
    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(unlabeled_df[features])
    
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    normalized_shap = mean_shap_values / np.sum(mean_shap_values)
    
    shap_FI = {"explainer": "shap"}
    for k, v in zip(unlabeled_df[features], normalized_shap):
        shap_FI[k] = v
    
    # Create SHAP plots
    shap.summary_plot(
        shap_values,
        unlabeled_df[features],
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"explanations/{sys.argv[1].split("/")[-1][:-4]}/shap_summary.png", dpi=300, bbox_inches="tight")
    plt.clf()
    
    #shap.plots.waterfall(
    #    shap.Explanation(
    #        values=shap_values[idx],
    #        base_values=explainer.expected_value,
    #        data=unlabeled_df[features].iloc[idx]
    #    ),
    #    show=False
    #)
    #plt.tight_layout()
    #plt.savefig(f"explanations/{sys.argv[1].split("/")[-1][:-4]}/shap_waterfall_{idx}.png", dpi=300, bbox_inches="tight")
    #plt.clf()
    
    return shap_FI, time.time()-t1

def analyze_feature_importance(feature_importance, features):
    """Analyze and visualize feature importance across explainers"""
    print("-----Feature Importance:-----")
    print(feature_importance)
    plt.figure(figsize=(15, 8))
    barWidth = 0.25
    r1 = np.arange(len(features))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    explainers = feature_importance['explainer'].unique()
    for i, explainer in enumerate(explainers):
        data = feature_importance[feature_importance['explainer'] == explainer]
        values = data[features].values[0]
        plt.bar([r1, r2, r3][i], values, width=barWidth, label=explainer)
    
    plt.xlabel('Features', fontweight='bold')
    plt.ylabel('Feature Importance', fontweight='bold')
    plt.title('Feature Importance Comparison Across Different Explainers')
    plt.xticks([r + barWidth for r in range(len(features))], features, rotation=45, ha='right')
    plt.ylim([0.0, 1.0])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(f'explanations/{sys.argv[1].split("/")[-1][:-4]}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_features(feature_importance, k):
    #print("\nTop half features for each explainer:")
    top_features = {}
    for _, row in feature_importance.iterrows():
        explainer = row['explainer']
        feature_values = pd.to_numeric(row.drop('explainer'), errors='coerce')
        top_2 = feature_values.nlargest(k)
        top_features[explainer] = [i for i in top_2.index]
    return top_features

def coerce2(s,cols):
  try: return int(s)
  except Exception:
    try: return float(s)
    except Exception:
      s = s.strip()
      if s not in cols and s != "?":    
          s += "X"
      return True if s=="True" else (False if s=="False" else s)

def csv2(file, cols):
  with open(sys.stdin if file=="-" else file, encoding="utf-8") as src:
    for line in src:
      line = re.sub(r'([\n\t\r ]|#.*)', '', line)
      if line: yield [coerce2(s, cols) for s in line.split(",")]

def regression(unlabeled, labeled, cols, data, regressor, top_pick):
    features = [c for c in cols if c[-1] not in ["+","-", "X"]]
    labeled_df = pd.DataFrame(labeled, columns=cols)
    unlabeled_df = pd.DataFrame(unlabeled, columns=cols)
    labeled_y = pd.DataFrame([ydist(row, data) for row in labeled], columns=["d2h"])
    unlabeled_y = pd.DataFrame([ydist(row, data) for row in unlabeled], columns=["d2h"])

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

    if regressor == "lgbm":
        params = {
        'objective': 'regression',  # For regression tasks
        'metric': 'mape',           # Root Mean Squared Error
        'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
        'learning_rate': 0.1,       # Learning rate
        'num_leaves': 31,           # Number of leaves in one tree
        'min_data_in_leaf': 1,     # Minimum number of data in a leaf
        'verbose': -1               # Suppress warning messages
        }
        train_data = lgb.Dataset(labeled_df[features], label=labeled_y)
        gbm = lgb.train(params, train_data, num_boost_round=100)
        predict = gbm.predict(unlabeled_df[features], num_iteration=gbm.best_iteration)
    elif regressor == "linear":
        lr = LinearRegression()
        lr.fit(labeled_df[features], labeled_y)
        predict = [i[0] for i in lr.predict(unlabeled_df[features]) ]
    elif regressor == "rf":
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(labeled_df[features], labeled_y.values.ravel())
        predict = rf.predict(unlabeled_df[features])    
    elif regressor == "svr":
        svr = SupportVectorRegressor(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        svr.fit(labeled_df[features], labeled_y.values.ravel())
        predict = svr.predict(unlabeled_df[features])
    elif regressor == "ann":
        predict = neural_net(unlabeled, labeled, cols, data, top_pick)

    top_idx = np.argsort(predict)[:top_pick]
    #print("features:", features)
    #for i in range(len(predict)):
    #    print(predict[i], unlabeled_y.iloc[i].values)
    #print(sorted([unlabeled_y.iloc[i].values for i in top_idx]))
    #input()
    return sorted([unlabeled_y.iloc[i].values for i in top_idx])[-1][0] ## Returning d2h of worst point

def exp1(file, columns, repeats, regressor, top_pick = 5, stop = 32):
    def win(x): return round(100*(1 - (x - b4.lo)/(b4.mu - b4.lo)))
    selected_raw_data = Data(csv2(file, columns))
    data, test_data = split_data(selected_raw_data)
    b4    = yNums(data.rows, selected_raw_data)
    the.Stop = stop
    stats = []
    for _ in range(repeats):
        #model = actLearn(data, shuffle=True)
        #labeled = model.best.rows + model.rest.rows
        data, test_data = split_data(selected_raw_data)
        labeled = data.rows
        unlabeled = test_data.rows
        stats.append( win( regression(unlabeled, labeled, [d.txt for d in data.cols.all], selected_raw_data, regressor, top_pick) ) )
    return np.mean(stats), np.std(stats)

def exp2(file, columns, repeats, top_pick = 5, stop = 32):
    def win(x): return round(100*(1 - (x - b4.lo)/(b4.mu - b4.lo)))
    selected_raw_data = Data(csv2(file, columns))
    data, test_data = split_data(selected_raw_data)
    b4    = yNums(data.rows, selected_raw_data)
    unlabeled_y = pd.DataFrame([ydist(row, data) for row in test_data.rows], columns=["d2h"])
    the.Stop = stop
    stats = []
    for _ in range(repeats):
        data, test_data = split_data(selected_raw_data)
        unlabeled_y = pd.DataFrame([ydist(row, data) for row in test_data.rows], columns=["d2h"])
        model = actLearn(data,shuffle=False)
        nodes = tree(model.best.rows + model.rest.rows,data)
        guesses = sorted([(leaf(nodes,row).ys, i) for i, row in enumerate(test_data.rows)],key=first)
        stats.append( win(sorted([unlabeled_y.iloc[i].values for _, i in guesses[:top_pick]])[-1][0]) )
    return np.mean(stats), np.std(stats)

def find_optimal(file, picks):
    def win(x): return round(100*(1 - (x - b4.lo)/(b4.mu - b4.lo)))
    selected_raw_data = Data(csv(file))
    data, test_data = split_data(selected_raw_data)
    b4    = yNums(data.rows, selected_raw_data)
    unlabeled_y = pd.DataFrame([ydist(row, data) for row in test_data.rows], columns=["d2h"])
    return win(sorted(unlabeled_y["d2h"])[picks-1]), win(sorted(unlabeled_y["d2h"])[int(len(test_data.rows)*0.1)])

def reliefff(data, cols):
    t1 = time.time()
    numerical_cols = [f for f in cols if (f[0].isupper() and f[-1] not in ["+","-", "X"])]
    categorical_cols = [f for f in cols if (not f[0].isupper() and f[-1] not in [["+","-", "X"]])]
    sample_rows = random.sample(data.rows, 100)
    X_train = pd.DataFrame(sample_rows, columns=cols)
    X_train.drop([c for c in cols if c[-1] in ["+","-", "X"]], axis=1, inplace=True)
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OrdinalEncoder(), categorical_cols)
    ])
    X_preprocessed = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    relief =  RReliefF(X_preprocessed, np.array([ydist(row, data) for row in sample_rows]), k=10, sigma=50)

    features = [c for c in cols if c[-1] not in ["+", "-", "X"]]
    weights = [f for f in relief]
    w = sum(weights)
    rlf = {"explainer":"rlf"}
    for f, v in zip(features, weights):
        rlf[f] = v[0] / w[0]
    return rlf, time.time()-t1
    
def main():
    dataset = sys.argv[1]
    raw_data = Data(csv(dataset))
    data, test_data = split_data(raw_data)

    try:
        os.makedirs(f"explanations/{dataset.split("/")[-1][:-4]}", exist_ok=True)
        print(f"Directory created successfully")
    except Exception as e:
        print(f"Error creating directory: {e}")
    
    cols = [d.txt for d in data.cols.all]
    targets = [c for c in cols if c[-1] in ["+", "-"]]
    features = [c for c in cols if c[-1] not in ["+", "-", "X"]]
    
    distribution_plot(data.rows, data, features, "all")
    model = actLearn(data)
    distribution_plot(model.best.rows + model.rest.rows, data, features, "32")
    
    # Run all explainers
    bl_FI, bl_time = run_bl_explainer(data, test_data, features)
    #lime_FI = run_lime_explainer(data, test_data, features)
    shap_FI, shap_time = run_shap_explainer(data, test_data, features)
    rlf_FI, rlf_time = reliefff(data, cols)
    
    # Combine results
    feature_importance = pd.DataFrame(columns=["explainer"] + features)
    feature_importance.loc[len(feature_importance)] = bl_FI
    #feature_importance.loc[len(feature_importance)] = lime_FI
    feature_importance.loc[len(feature_importance)] = shap_FI
    feature_importance.loc[len(feature_importance)] = rlf_FI
    
    analyze_feature_importance(feature_importance, features)
    for regressor in ["bl", "ann","rf", "svr", "lgbm","linear"]:
        bl_mean, bl_std = [], []
        shap_mean, shap_std = [], []
        rlf_mean, rlf_std = [], []
        t1 = time.time()
        for k in range(len(features)):
            top_features = get_features(feature_importance, k+1)
            if regressor == "bl":
                mean, std = exp2(dataset, top_features['BL'] + targets, 15, 5, 32)
                bl_mean.append(mean)
                bl_std.append(std)
                mean, std = exp2(dataset, top_features['shap'] + targets, 15, 5, 32)
                shap_mean.append(mean)
                shap_std.append(std)
                mean, std = exp2(dataset, top_features['rlf'] + targets, 15, 5, 32)
                rlf_mean.append(mean)
                rlf_std.append(std)
            else:
                mean, std = exp1(dataset, top_features['BL'] + targets, 15, regressor, 5, 32)
                bl_mean.append(mean)
                bl_std.append(std)
                mean, std = exp1(dataset, top_features['shap'] + targets, 15, regressor, 5, 32)
                shap_mean.append(mean)
                shap_std.append(std)
                mean, std = exp1(dataset, top_features['rlf'] + targets, 15, regressor, 5, 32)
                rlf_mean.append(mean)
                rlf_std.append(std)
                
        optimal, optimal_90 = find_optimal(dataset, 5)
        #print(bl_mean,"\n", shap_mean, optimal)
        # Create the plot
        plt.figure(figsize=(10, 6))
        x = range(1, len(features)+1)
        plt.errorbar(x, bl_mean, yerr=bl_std, label=f'BL({round(bl_time,3)})', marker='o', capsize=5)
        plt.errorbar([xx+0.01 for xx in x], shap_mean, yerr=shap_std, label=f'SHAP({round(shap_time,3)})', marker='s', capsize=5)
        plt.errorbar([xx+0.02 for xx in x], rlf_mean, yerr=rlf_std, label=f'ReliefF({round(rlf_time,3)})', marker='s', capsize=5)

        
        plt.errorbar(x, [optimal for _ in range(len(bl_mean))], label = f"Optimal 5/{len(test_data.rows)}", marker='o', capsize=5)
        plt.errorbar(x, [optimal_90 for _ in range(len(bl_mean))], label = f"90% Optimal {int(0.1 * len(test_data.rows))}/{len(test_data.rows)}", marker='o', capsize=5)
        # Customize the plot
        plt.xlabel('# Top Features')
        plt.ylabel('% worst optimal d2hs')
        plt.title(f'Performance Comparison (regresssion: {regressor}, {(time.time()-t1)//0.1}s)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.ylim(-100, 100)
        plt.xticks(range(1, len(features)+1))
        
        # Save the plot
        plt.savefig(f'explanations/{dataset.split("/")[-1][:-4]}/{regressor}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Done with {regressor}.")

if __name__ == "__main__":
    main()
