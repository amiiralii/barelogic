import os
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_csv_file(file_path):
    results = []
    chck = False
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if "asIs," in line: line = line.replace("asIs,", "asIs,,")
            if not line or line == '#':
                chck = True
                continue
            words = line.split(",")
            
            if chck:
                treatment = {}
                treatment['rank'] = int(words[0]) * -1
                treatment['trt'] = words[2].strip() + "_" + words[1].strip()
                treatment['mean'] = words[3]
                results.append(treatment)
    print(file_path)
    rank_stabilizer = min([i['rank'] for i in results]) * -1
    for i in results:
        i['rank'] = i['rank'] + rank_stabilizer
    return results

def analyze_results():
    results_dir = Path('results_final_1/')
    if not results_dir.exists():
        print(f"Warning: {results_dir} directory does not exist")
        return {}
    cols, cols2 = [], []
    for feature_selection in [ "RLF", "SHAP", "BL", "anova", "all"]:
        for regressor in ["linear", "rf", "svr", "ann", "lgbm", "bl"]:
            cols.append(f"{feature_selection}_{regressor}")
    cols.append('_asIs')
    cols.append('data')
    for feature_selection in [ "RLF", "SHAP", "BL", "anova", "all"]:
        for regressor in ["linear", "rf", "svr", "ann", "lgbm", "bl"]:
            cols2.append(f"{feature_selection}_{regressor}")
            cols2.append(f"{feature_selection}_{regressor}_rank")
    cols2.append('_asIs')
    cols2.append('_asIs_rank')
    cols2.append('data')
    results_df = pd.DataFrame(columns=cols)
    colors_df = pd.DataFrame(columns=cols)
    all_df = pd.DataFrame(columns=cols2)
    for file_path in results_dir.glob('*.csv'):
        res = process_csv_file(file_path)
        new_row = {}
        color_row = {}
        all_row = {}
        for i in res:
            new_row[i['trt']] = i['mean']
            color_row[i['trt']] = i['rank']

            all_row[f"{i['trt']}_rank"] = i['rank']
            all_row[f"{i['trt']}"] = i['mean']
        new_row['data'] = file_path.name.split('/')[-1].split('.')[0]
        color_row['data'] = file_path.name.split('/')[-1].split('.')[0]
        all_row['data'] = file_path.name.split('/')[-1].split('.')[0]
        results_df.loc[len(results_df)] = new_row
        colors_df.loc[len(colors_df)] = color_row
        all_df.loc[len(all_df)] = all_row
    return results_df, colors_df, all_df

if __name__ == "__main__":
    results, colors, all = analyze_results()
    
    #colors.to_csv('colors.csv', index=False)
    desired_order = [ 'RLF_linear', 'RLF_linear_rank', 'SHAP_linear', 'SHAP_linear_rank', 'BL_linear', 'BL_linear_rank', 'anova_linear', 'anova_linear_rank', 'all_linear', 'all_linear_rank',
                     'RLF_rf', 'RLF_rf_rank', 'SHAP_rf', 'SHAP_rf_rank', 'BL_rf', 'BL_rf_rank', 'anova_rf', 'anova_rf_rank', 'all_rf', 'all_rf_rank',
                     'RLF_svr', 'RLF_svr_rank', 'SHAP_svr', 'SHAP_svr_rank', 'BL_svr', 'BL_svr_rank', 'anova_svr', 'anova_svr_rank', 'all_svr', 'all_svr_rank',
                     'RLF_ann', 'RLF_ann_rank', 'SHAP_ann', 'SHAP_ann_rank', 'BL_ann', 'BL_ann_rank', 'anova_ann', 'anova_ann_rank', 'all_ann', 'all_ann_rank',
                     'RLF_lgbm', 'RLF_lgbm_rank', 'SHAP_lgbm', 'SHAP_lgbm_rank', 'BL_lgbm', 'BL_lgbm_rank', 'anova_lgbm', 'anova_lgbm_rank', 'all_lgbm', 'all_lgbm_rank',
                     'RLF_bl', 'RLF_bl_rank', 'SHAP_bl', 'SHAP_bl_rank', 'BL_bl', 'BL_bl_rank', 'anova_bl', 'anova_bl_rank', 'all_bl', 'all_bl_rank',
        '_asIs', '_asIs_rank', 'data']
    all[desired_order].to_csv('combined_bigs.csv', index=False)
    desired_order2 = [ 'RLF_linear', 'SHAP_linear', 'BL_linear', 'anova_linear', 'all_linear',
                     'RLF_rf', 'SHAP_rf', 'BL_rf', 'anova_rf', 'all_rf',
                     'RLF_svr', 'SHAP_svr', 'BL_svr', 'anova_svr', 'all_svr',
                     'RLF_ann', 'SHAP_ann', 'BL_ann', 'anova_ann', 'all_ann',
                     'RLF_lgbm', 'SHAP_lgbm', 'BL_lgbm', 'anova_lgbm', 'all_lgbm',
                     'RLF_bl', 'SHAP_bl', 'BL_bl', 'anova_bl', 'all_bl',
        '_asIs', 'data']
    results[desired_order2].to_csv('results.csv', index=False)
