import os
import csv
from pathlib import Path
import pandas as pd

def process_csv_file(file_path):
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line == '#':
                continue
            words = line.split(",")
            if len(words) > 2:
                treatment = {}
                treatment['rank'] = words[0]
                treatment['trt'] = words[2].strip() + "_" + words[1].strip()
                treatment['mean'] = words[3]
                results.append(treatment)
    return results

def analyze_results():
    results_dir = Path('results')
    if not results_dir.exists():
        print(f"Warning: {results_dir} directory does not exist")
        return {}
    cols = []
    for feature_selection in [ "RLF", "SHAP", "BL"]:
        for regressor in ["linear", "rf", "svr", "ann", "lgbm", "bl"]:
            cols.append(f"{feature_selection}_{regressor}")
    cols.append('data')
    results_df = pd.DataFrame(columns=cols)
    colors_df = pd.DataFrame(columns=cols)
    for file_path in results_dir.glob('*.csv'):
        res = process_csv_file(file_path)
        new_row = {}
        color_row = {}
        for i in res:
            new_row[i['trt']] = i['mean']
            color_row[i['trt']] = i['rank']
        new_row['data'] = file_path.name.split('/')[-1].split('.')[0]
        color_row['data'] = file_path.name.split('/')[-1].split('.')[0]
        results_df.loc[len(results_df)] = new_row
        colors_df.loc[len(colors_df)] = color_row
    return results_df, colors_df

if __name__ == "__main__":
    results, colors = analyze_results()
    results.to_csv('results.csv', index=False)
    colors.to_csv('colors.csv', index=False)
    