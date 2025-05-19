from bl import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import matplotlib.pyplot as plt
import random
import lime
import lime.lime_tabular
import shap
from sklearn.preprocessing import StandardScaler
import os

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

def explain_predictions(model, data, regressor="lgbm"):
    # Prepare data for explanations
    features = [c.txt for c in data.cols.all if c.txt[-1] not in ["+","-", "X"]]
    targets = [c.txt for c in data.cols.all if c.txt[-1] in ["+","-"]]
    
    # Convert data to DataFrame
    df = pd.DataFrame(data.rows, columns=[c.txt for c in data.cols.all])
    
    # Prepare feature data
    X = df[features]
    y = df[targets]  # Get all target values
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    
    # MDI (Mean Decrease in Impurity) Explanation
    def mdi_explanation():
        if regressor == "lgbm":
            # Get feature importance from the model
            importance = model.feature_importance(importance_type='gain')
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': importance
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.bar(feature_importance['feature'], feature_importance['importance'])
            plt.xticks(rotation=45, ha='right')
            plt.title('Feature Importance (MDI)')
            plt.tight_layout()
            plt.savefig("explanations/mdi_importance.png", bbox_inches='tight', dpi=300)
            plt.close()
            
            # Save feature importance to CSV
            feature_importance.to_csv("explanations/mdi_importance.csv", index=False)
    
    # LIME Explanation
    def lime_explanation():
        # Create explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_scaled,
            feature_names=features,
            class_names=targets,
            mode='regression'
        )
        
        # Explain a few examples and save to file
        for i in range(min(5, len(X))):
            # Get actual target values for this instance
            actual_values = y.iloc[i]
            
            # Create explanation for each target
            for target_idx, target in enumerate(targets):
                if regressor == "lgbm":
                    # For LightGBM, we need to handle the multi-target case
                    def predict_fn(x):
                        preds = model.predict(x)
                        if len(preds.shape) > 1:
                            return preds[:, target_idx]
                        return preds
                else:
                    # For linear regression, we'll use the model directly
                    def predict_fn(x):
                        return model.predict(x)
                
                try:
                    exp = explainer.explain_instance(
                        X_scaled[i], 
                        predict_fn,
                        num_features=len(features)
                    )
                    
                    # Add actual target value to the explanation
                    exp_html = exp.as_html()
                    exp_html = exp_html.replace(
                        '<div class="prediction">',
                        f'<div class="prediction"><p>Actual {target} value: {actual_values[target]:.4f}</p>'
                    )
                    
                    # Save to file with target name
                    with open(f"explanations/lime_explanation_{i}_{target}.html", 'w') as f:
                        f.write(exp_html)
                except Exception as e:
                    print(f"Error explaining instance {i} for target {target}: {str(e)}")
                    print(f"Model type: {regressor}")
                    print(f"Prediction shape: {model.predict(X_scaled[i:i+1]).shape}")
                    continue
            
    # SHAP Explanation
    def shap_explanation():
        if regressor == "lgbm":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
        else:
            explainer = shap.KernelExplainer(model.predict, X_scaled_df[:100])
            shap_values = explainer.shap_values(X_scaled_df)
                
        # Save summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_scaled_df)
        plt.savefig("explanations/summary_plot.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save dependence plots for top features
        for feature in features[:3]:
            plt.figure()
            shap.dependence_plot(
                feature,
                shap_values, 
                X_scaled_df,
                feature_names=features
            )
            plt.savefig(f"explanations/dependence_plot_{feature}.png", bbox_inches='tight', dpi=300)
            plt.close()
    
    return mdi_explanation, lime_explanation, shap_explanation

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
            
            # Add explanations
            mdi_expl, lime_expl, shap_expl = explain_predictions(gbm, data, "lgbm")
            mdi_expl()  # Generate MDI explanation
            lime_expl()  # Generate LIME explanations
            shap_expl()  # Generate SHAP explanations
            
        elif regressor == "linear":
            model = LinearRegression()
            model.fit(labeled_df[features], labeled_df[t])
            predict = model.predict(unlabeled_df[features])
            
            # Add explanations
            mdi_expl, lime_expl, shap_expl = explain_predictions(model, data, "linear")
            mdi_expl()  # Generate MDI explanation
            lime_expl()  # Generate LIME explanations
            shap_expl()  # Generate SHAP explanations
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
    the.Stop = 32

    stats = []
    for _ in range(repeats):
        model = actLearn(data, shuffle=True)
        labeled = model.best.rows + model.rest.rows
        unlabeled = test_data.rows
        stats.append(lightgbm(unlabeled, labeled, [d.txt for d in data.cols.all], raw_data, stats, regressor))
    mean = sum(s[0] for s in stats) / len(stats)
    std = sum(s[1] for s in stats) / len(stats)

    return mean, std

def exp2(file, repeats):
    raw_data  = Data(csv(file))
    data, test_data = split_data(raw_data)
    
    out = {str(j):[] for j in [512,256,128,64,32,16,8]}
    the.Stop = 32
    stat = []
    for _ in range(repeats):
        model = actLearn(data,shuffle=True)
        nodes = tree(model.best.rows + model.rest.rows,data)
        guesses = sorted([(leaf(nodes,row).ys,row) for row in test_data.rows],key=first)
        d2h_results = [ydist(guess,data) for _,guess in guesses[:5]]
        stat.append([np.mean(d2h_results), np.std(d2h_results)])
    mean = sum(s[0] for s in stat) / len(stat)
    std = sum(s[1] for s in stat) / len(stat)
    return mean, std
    


dataset = sys.argv[1]
t1 = time.time()
results1 = exp1(file=dataset, repeats=1)
lgbm_time = round(time.time() - t1,3)
print(f'--- {lgbm_time}s ---')
t2 = time.time()
results2 = exp1(file=dataset, repeats=1, regressor="linear")
lr_time = round(time.time() - t2,3)
print(f'--- {lr_time}s ---')
t3 = time.time()
results3 = exp2(file=dataset, repeats=1)
bl_time = round(time.time() - t3,3)
print(f'--- {bl_time}s ---')
