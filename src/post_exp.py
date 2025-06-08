import os
import csv
from pathlib import Path

def process_csv_file(file_path):
    """
    Process a single CSV file and return a dictionary where:
    - Keys are the first word of each line
    - Values are lists containing the rest of the words in the line
    - Lines containing only '#' are ignored
    """
    result_dict = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines and lines containing only '#'
            line = line.strip()
            if not line or line == '#':
                continue
                
            # Split the line into words
            words = line.split(",")
            if words:  # Make sure we have at least one word
                key = words[0]
                values = words[1:]  # All words after the first one
                result_dict[key] = values
    for i in result_dict:
        print(i, result_dict[i])
    return result_dict

def analyze_results():
    """
    Analyze all CSV files in the results directory
    """
    results_dir = Path('results')
    
    # Check if results directory exists
    if not results_dir.exists():
        print(f"Warning: {results_dir} directory does not exist")
        return {}
    
    # Dictionary to store results from all files
    all_results = {}
    
    # Process each CSV file in the results directory
    for file_path in results_dir.glob('*.csv'):
        print(f"Processing file: {file_path}")
        file_results = process_csv_file(file_path)
        all_results[file_path.name] = file_results
        input()
    return all_results

if __name__ == "__main__":
    results = analyze_results()
    
    # Print results for verification
    for filename, file_data in results.items():
        print(f"\nResults from {filename}:")
        for key, values in file_data.items():
            print(f"{key}: {values}") 