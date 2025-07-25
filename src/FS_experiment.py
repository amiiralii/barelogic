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
import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


def split_data(data, r_seed = 42, test_size=0.2):
    # Shuffle the data
    rows = data.rows.copy()
    random.seed(r_seed)
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

def run_bl_explainer(data):
    """Run BareLogic explainer and return feature importance"""
    #print('-------BL:-------')
    t1 = time.time()
    stts = []
    count = 0
    rpt = 5
    for _ in range(rpt):
        model = actLearn(data, shuffle=True)
        nodes = tree(model.best.rows + model.rest.rows, data)
        vals = treeFeatureImportance(nodes)
        stts.append(vals)
        count += len([i for i in vals.values() if i > 0])
    print("BL Labeling Budget,", rpt*the.Stop)
    print("---------------------")
    bl_FI = {}
    for f in data.cols.x:
        bl_FI[f.txt] = np.average([stts[i][f.txt] for i in range(len(stts))])
    w = sum(vf for vf in bl_FI.values())
    for f in bl_FI:
        bl_FI[f] = bl_FI[f] / w
    bl_FI["explainer"] = "BL"
    bl_FI["run_time"] = (time.time()-t1)
    return bl_FI, round( count / rpt )

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

def run_anova_explainer(data, features):
    """Run ANOVA explainer and return feature importance"""
    labeled = data.rows
    cols = [d.txt for d in data.cols.all]
    labeled_df = pd.DataFrame(labeled, columns=cols)
    le = LabelEncoder()
    for c in cols:
        if c not in features: labeled_df.drop(c, axis=1, inplace=True) 
        elif c[0].isupper():
            labeled_df[c] = pd.to_numeric(labeled_df[c], errors='coerce')
        else:
            labeled_df[c] = labeled_df[c].astype('category')
            labeled_df[c] = le.fit_transform(labeled_df[c])
    labeled_df['d2h'] = pd.DataFrame([ydist(row, data) for row in labeled], columns=["d2h"])

    t1 = time.time()

    # Fit a regression model using all features
    formula = 'd2h' + ' ~ ' + ' + '.join(features)
    reg = ols(formula, data=labeled_df).fit()

    # Perform Type II ANOVA
    aov_table = sm.stats.anova_lm(reg, typ=2)
    aov_table.sort_values(by='sum_sq', ascending=False, inplace=True)
    aov_table['importance'] = aov_table['sum_sq'] / aov_table['sum_sq'].sum()
    aov_table.drop('Residual', errors='ignore', inplace=True)

    anova_FI = {}
    for index, row in aov_table.iterrows():
        anova_FI[index] = row['importance']
    anova_FI["explainer"] = "anova"
    anova_FI["run_time"] = (time.time()-t1) / 15
    return anova_FI

def run_shap_explainer(data, test_data, features, idx=5):
    """Run SHAP explainer and return feature importance"""
    #print('-------SHAP:-------')
    t1=time.time()
    # Prepare data
    labeled = data.rows
    unlabeled = test_data.rows
    cols = [d.txt for d in data.cols.all]
    
    labeled_df = pd.DataFrame(labeled, columns=cols)
    unlabeled_df = pd.DataFrame(unlabeled, columns=cols)

    # Preprocess data
    le = LabelEncoder()
    for c in cols:
        if c not in features: labeled_df.drop(c, axis=1, inplace=True) 
        elif c[0].isupper():
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
    
    labeled_y = pd.DataFrame([ydist(row, data) for row in labeled], columns=["d2h"])
    model.fit(labeled_df[features], labeled_y)
    
    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(unlabeled_df[features])
    
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    normalized_shap = mean_shap_values / np.sum(mean_shap_values)
    
    shap_FI = {"explainer": "shap", "run_time":time.time()-t1}
    for k, v in zip(unlabeled_df[features], normalized_shap):
        shap_FI[k] = v
    
    # Create SHAP plots
    #shap.summary_plot(
    #    shap_values,
    #    unlabeled_df[features],
    #    plot_type="bar",
    #    show=False
    #)
    #plt.tight_layout()
    #plt.savefig(f"explanations/{sys.argv[1].split("/")[-1][:-4]}/shap_summary.png", dpi=300, bbox_inches="tight")
    #plt.clf()
    
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
    return shap_FI

def analyze_feature_importance(feature_importance, features):
    """Analyze and visualize feature importance across explainers"""
    #print("-----Feature Importance:-----")
    #print(feature_importance)
    plt.figure(figsize=(15, 8))
    barWidth = 0.25
    r1 = np.arange(len(features))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    explainers = feature_importance['explainer'].unique()
    for i, explainer in enumerate(explainers):
        data = feature_importance[feature_importance['explainer'] == explainer]
        values = data[features].values[0]
    #    plt.bar([r1, r2, r3][i], values, width=barWidth, label=f"{explainer} ({round(data['run_time'].values[0],3)} s)")
    
    #plt.xlabel('Features', fontweight='bold')
    #plt.ylabel('Feature Importance', fontweight='bold')
    #plt.title('Feature Importance Comparison Across Different Explainers (Total Time)')
    #plt.xticks([r + barWidth for r in range(len(features))], features, rotation=45, ha='right')
    #plt.ylim([0.0, 1.0])
    #plt.legend()
    #plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    #plt.tight_layout()
    #plt.savefig(f'explanations/{sys.argv[1].split("/")[-1][:-4]}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    #plt.close()

def get_features(feature_importance, k):
    #print("\nTop half features for each explainer:")
    top_features = {}
    for _, row in feature_importance.iterrows():
        explainer = row['explainer']
        feature_values = pd.to_numeric(row.drop(['explainer','run_time']), errors='coerce')
        top_k = feature_values.nlargest(k)
        top_features[explainer] = [i for i in top_k.index]
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

def regression(labeled_df, unlabeled_df, labeled_y, unlabeled_y, cols, regressor, top_pick):
    features = [c for c in cols if c[-1] not in ["+","-", "X"]]
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
        'verbose': -1              # Suppress warning messages
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
        predict = neural_net(unlabeled_df, unlabeled_y, labeled_df, labeled_y, cols)

    top_idx = np.argsort(predict)[:top_pick]
    #print("features:", features)
    #for i in range(len(predict)):
    #    print(predict[i], unlabeled_y.iloc[i].values)
    #print(sorted([unlabeled_y.iloc[i].values for i in top_idx]))
    #input()
    return sorted([unlabeled_y.iloc[i].values for i in top_idx])[0][0]

def exp1(selected_raw_data, data, test_data, b4, regressor, top_pick = 5):
    def win(x): return round(100*(1 - (x - b4.lo)/(b4.mu - b4.lo)))
    labeled = data.rows
    unlabeled = test_data.rows
    cols = [d.txt for d in data.cols.all]
    labeled_df = pd.DataFrame(labeled, columns=cols)
    unlabeled_df = pd.DataFrame(unlabeled, columns=cols)
    labeled_y = pd.DataFrame([ydist(row, data) for row in labeled], columns=["d2h"])
    unlabeled_y = pd.DataFrame([ydist(row, selected_raw_data) for row in unlabeled], columns=["d2h"])

    return win( regression(labeled_df, unlabeled_df, labeled_y, unlabeled_y, cols, regressor, top_pick) )

def exp2(selected_raw_data, data, test_data, b4, top_pick = 5, stop = 32):
    def win(x): return round(100*(1 - (x - b4.lo)/(b4.mu - b4.lo)))
    unlabeled_y = pd.DataFrame([ydist(row, selected_raw_data) for row in test_data.rows], columns=["d2h"])
    the.Stop = stop
    model = actLearn(data,shuffle=False)
    nodes = tree(model.best.rows + model.rest.rows,data)
    guesses = sorted([(leaf(nodes,row).ys, i) for i, row in enumerate(test_data.rows)],key=first)
    acc = win(sorted([unlabeled_y.iloc[i].values for _, i in guesses[:top_pick]])[0][0])
    vals = treeFeatureImportance(nodes)
    return acc, len([i for i in vals.values() if i > 0])

def find_optimal(selected_raw_data, test_data, b4, picks):
    def win(x): return round(100*(1 - (x - b4.lo)/(b4.mu - b4.lo)))
    unlabeled_y = pd.DataFrame([ydist(row, selected_raw_data) for row in test_data.rows], columns=["d2h"])
    return win(sorted(unlabeled_y["d2h"])[picks-1]), win(sorted(unlabeled_y["d2h"])[int(len(test_data.rows)*0.1)])

def reliefff(data, cols):
    #print('-------ReliefF:-------')
    t1 = time.time()
    X_train = pd.DataFrame(data.rows, columns=cols)
    X_train.drop([c for c in cols if c[-1] in ["+","-", "X"]], axis=1, inplace=True)
    
    # Get columns after dropping
    remaining_cols = X_train.columns.tolist()
    numerical_cols = [f for f in remaining_cols if f[0].isupper()]
    categorical_cols = [f for f in remaining_cols if not f[0].isupper()]
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OrdinalEncoder(), categorical_cols)
    ], sparse_threshold=0)
    X_preprocessed = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    if len(data.rows) < 5000: sample_size = len(data.rows)
    elif len(data.rows) < 20002: sample_size = 5000
    else: sample_size = 10000
    relief =  RReliefF(X_preprocessed, np.array([ydist(row, data) for row in data.rows]), updates = sample_size,k=10, sigma=50)

    features = [c for c in cols if c[-1] not in ["+", "-", "X"]]
    weights = [f[0] if f > 0 else 0 for f in relief]
    w = sum(weights)
    if w == 0: 
        tmp = min(relief)
        weights = [f[0] + tmp for f in relief]
        w = sum(weights)
    rlf = {"explainer":"rlf", "run_time":time.time()-t1}
    for f, v in zip(features, weights):
        rlf[f] = v / w
    return rlf
    
def main():
    random_seed = 42
    dataset = sys.argv[1]
    raw_data = Data(csv(dataset))
    data, test_data = split_data(raw_data)
    stp = min(30, len(data.rows) // 10)
    the.Stop = stp
    the.acq = "xploit"

    cols = [d.txt for d in data.cols.all]
    targets = [c for c in cols if c[-1] in ["+", "-"]]
    features = [c for c in cols if c[-1] not in ["+", "-", "X"]]
    
    
    # Run all explainers
    bl_FI, feature_counts = run_bl_explainer(data)
    anova_FI = run_anova_explainer(data, features)
    shap_FI = run_shap_explainer(data, test_data, features)
    rlf_FI = reliefff(data, cols)
    
    for fi in [bl_FI, anova_FI, shap_FI, rlf_FI]:
        print(f"{fi['explainer']} time,\t{round(fi['run_time'], 3)}")

    print("---------------------")
    print(f"Number of features selected by BL, {feature_counts}")

    # Combine results
    feature_importance = pd.DataFrame(columns=["explainer", "run_time"] + features)
    feature_importance.loc[len(feature_importance)] = bl_FI
    feature_importance.loc[len(feature_importance)] = shap_FI
    feature_importance.loc[len(feature_importance)] = anova_FI
    feature_importance.loc[len(feature_importance)] = rlf_FI
    analyze_feature_importance(feature_importance, features)

    selected_features = get_features(feature_importance, feature_counts)
    for method in selected_features:
        print(f"{method} features,\t{', '.join(selected_features[method])}")
    print("---------------------")

    repeats = 20
    top_features = get_features(feature_importance, feature_counts)
    records = []
    for regressor in ["linear", "rf", "svr", "ann", "lgbm"]:
        bl, shap, rlf, all, anova = [], [], [], [], []
        t1 = time.time()
        for ii in range(repeats):
            random_seed *= 2
            selected_raw_data = Data(csv2(dataset, top_features['BL'] + targets))
            data, test_data = split_data(selected_raw_data, random_seed)
            b4    = yNums(data.rows, selected_raw_data)
            bl.append( exp1(selected_raw_data, data, test_data, b4, regressor, 10) )

            selected_raw_data = Data(csv2(dataset, top_features['shap'] + targets))
            data, test_data = split_data(selected_raw_data, random_seed)
            b4    = yNums(data.rows, selected_raw_data)
            shap.append( exp1(selected_raw_data, data, test_data, b4, regressor, 10) )

            selected_raw_data = Data(csv2(dataset, top_features['rlf'] + targets))
            data, test_data = split_data(selected_raw_data, random_seed)
            b4    = yNums(data.rows, selected_raw_data)
            rlf.append( exp1(selected_raw_data, data, test_data, b4, regressor, 10) )

            selected_raw_data = Data(csv2(dataset, top_features['anova'] + targets))
            data, test_data = split_data(selected_raw_data, random_seed)
            b4    = yNums(data.rows, selected_raw_data)
            anova.append( exp1(selected_raw_data, data, test_data, b4, regressor, 10) )

            selected_raw_data = Data(csv(dataset))
            data, test_data = split_data(selected_raw_data, random_seed)
            b4    = yNums(data.rows, selected_raw_data)
            all.append( exp1(selected_raw_data, data, test_data, b4, regressor, 10) )
        
        print(f"{regressor} time, {round(time.time()-t1, 3)}")
        bl_acc, shap_acc, rlf_acc, all_acc, anova_acc = stats.SOME(txt=f"{regressor},BL"), stats.SOME(txt=f"{regressor},SHAP"), stats.SOME(txt=f"{regressor},RLF"), stats.SOME(txt=f"{regressor},all"), stats.SOME(txt=f"{regressor},anova")
        bl_acc.adds(bl)
        shap_acc.adds(shap)
        rlf_acc.adds(rlf)
        all_acc.adds(all)
        anova_acc.adds(anova)
        records.append(bl_acc)
        records.append(shap_acc)
        records.append(rlf_acc)
        records.append(all_acc)
        records.append(anova_acc)

    stats.report(records)        


if __name__ == "__main__":
    main()
