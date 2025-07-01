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
    rank_stabilizer = min([i['rank'] for i in results]) * -1
    for i in results:
        i['rank'] = i['rank'] + rank_stabilizer
    return results

def analyze_results():
    results_dir = Path('results/big')
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
    #results.to_csv('results.csv', index=False)
    #colors.to_csv('colors.csv', index=False)
    all.to_csv('combined_bigs.csv', index=False)
