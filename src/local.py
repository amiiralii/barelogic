import sys
from bl import *
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from extend4 import *
def run_lime_explainer(data, test_data, features, idx=5):
    """Run LIME explainer and return feature importance"""
    #print('-------LIME:-------')
    
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


dataset = sys.argv[1]
selected_raw_data = Data(csv(dataset))

data, test_data = split_data(selected_raw_data, 95)
print(run_lime_explainer(data, test_data, data.cols.names))
#model = actLearn(data,shuffle=True)
#nodes = tree(model.best.rows + model.rest.rows,data)
