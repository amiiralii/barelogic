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
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from IPython.display import display, HTML


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

dataset = sys.argv[1]

raw_data  = Data(csv(dataset))
data, test_data = split_data(raw_data)

cols = [d.txt for d in data.cols.all]
features = [c for c in cols if c[-1] not in ["+","-", "X"]]
targets = [c for c in cols if c[-1] in ["+","-"]]

model = actLearn(data,shuffle=True)
nodes = tree(model.best.rows + model.rest.rows,data)

showTree(nodes)
idx = 5     # sample test feed into explainers
#lf = leaf(nodes,test_data.rows[idx])
print("MDI of BL tree:\t", round(treeMDI(nodes),3))
bl_FI = treeFeatureImportance(nodes)
w = sum(vf for vf in bl_FI.values())
for f in bl_FI:
    bl_FI[f] = bl_FI[f] / w
bl_FI["explainer"] = "BL"
feature_importance = pd.DataFrame(columns= ["explainer"] + features)
feature_importance.loc[len(feature_importance)] = bl_FI

##--------------------------------

print('-------LIME:-------')


labeled = model.best.rows + model.rest.rows
unlabeled = test_data.rows

labeled_df = pd.DataFrame(labeled, columns=[c for c in cols])
unlabeled_df = pd.DataFrame(unlabeled, columns=[c for c in cols])

labeled_y = pd.DataFrame([ydist(row,data) for row in labeled],columns=["d2h"])
unlabeled_y = pd.DataFrame([ydist(row,data) for row in unlabeled],columns=["d2h"])

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


# 3. Train an LGBM regression model
model = LGBMRegressor(
    boosting_type    = 'gbdt',
    objective        = 'regression',
    num_leaves       = 8,
    max_depth        = 3,
    learning_rate    = 0.1,
    n_estimators     = 50,
    min_data_in_leaf = 2,
    min_data_in_bin  = 1,
    max_bin          = 15,
    subsample        = 0.8,
    subsample_freq   = 1,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    force_col_wise   = True,
    verbose          = -1,
    random_state     = 42
)
model.fit(labeled_df[features], labeled_y)

# 4. Set up the LIME explainer for tabular data
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data         = labeled_df[features].values,
    feature_names         = features,
    categorical_features = [c for c in features if not c[0].isupper()],
    mode                  = 'regression',
    discretize_continuous = True
)

# 5. Choose an instance to explain
instance = unlabeled_df[features].iloc[idx].values.reshape(1, -1)
true_value = unlabeled_y.iloc[idx]
predicted_value = model.predict(instance)[0]

print(f"True target = {true_value["d2h"]}, Model prediction = {predicted_value}")

# 6. Generate explanation
exp = explainer.explain_instance(
    data_row     = instance.flatten(),
    predict_fn   = model.predict,
    num_features = 6   # top 6 features
)

# 7. Print out the explanation
print("LIME Explain:")
for feature, weight in exp.as_list():
    direction = "↑" if weight > 0 else "↓"
    print(f"{feature:>20}: {weight:+.3f} ({direction})")

# 8. Optional HTML output or in-notebook visualization:
exp.save_to_file('explanations/lgbm_lime_explanation.html')

total_gain = model.booster_.feature_importance(importance_type='gain')

# 2. Compute MDI: mean gain per tree
n_trees = model.booster_.num_trees()
mdi = total_gain / n_trees

# 3. Organize into a DataFrame
mdi_df = pd.DataFrame({
    'feature': features,
    'total_gain': total_gain,
    'mdi': mdi
}).sort_values('mdi', ascending=False)

#print(mdi_df)
mdi_df['mdi_norm'] = mdi_df['mdi'] / mdi_df['mdi'].sum()
print(mdi_df[['feature','mdi_norm']])
lime_FI = {"explainer":"lime"}
for index, row in mdi_df.iterrows():
    lime_FI[row['feature']] = row['mdi_norm']
feature_importance.loc[len(feature_importance)] = lime_FI

##--------------------------------

print('-------SHAP:-------')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(unlabeled_df[features])

mean_shap_values = np.abs(shap_values).mean(axis=0)
normalized_shap = mean_shap_values / np.sum(mean_shap_values)

shap_FI = {"explainer":"shap"}
for k,v in zip(unlabeled_df[features], normalized_shap):
    shap_FI[k]=v
feature_importance.loc[len(feature_importance)] = shap_FI

shap.summary_plot(
    shap_values,
    unlabeled_df[features],
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig("explanations/shap_summary.png", dpi=300, bbox_inches="tight")
plt.clf()

print(f"True target = {unlabeled_y.iloc[idx]["d2h"]}, model prediction = {model.predict(unlabeled_df[features].iloc[[idx]])[0]}")
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=unlabeled_df[features].iloc[idx]
    ),
    show=False
)

plt.tight_layout()
plt.savefig("explanations/shap_waterfall_{}.png".format(idx), dpi=300, bbox_inches="tight")
plt.clf()


##--------------
print(feature_importance)

# Create grouped bar chart for feature importance
plt.figure(figsize=(15, 8))

# Set width of bars and positions of the bars
barWidth = 0.25
r1 = np.arange(len(features))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Create the bars
explainers = feature_importance['explainer'].unique()
for i, explainer in enumerate(explainers):
    data = feature_importance[feature_importance['explainer'] == explainer]
    values = data[features].values[0]
    plt.bar([r1, r2, r3][i], values, width=barWidth, label=explainer)

# Add labels and formatting
plt.xlabel('Features', fontweight='bold')
plt.ylabel('Feature Importance', fontweight='bold')
plt.title('Feature Importance Comparison Across Different Explainers')
plt.xticks([r + barWidth for r in range(len(features))], features, rotation=45, ha='right')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')
plt.tight_layout()
plt.savefig('explanations/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

