# ============================================
# PART 1: CHECK AND LOAD COMPLETE DATASET
# ============================================

import pandas as pd
import numpy as np
import os

print("=" * 50)
print("LOADING COMPLETE GERMAN CREDIT DATASET")
print("=" * 50)

# First, let's try to load from UCI repository
try:
    from ucimlrepo import fetch_ucirepo

    print("Fetching dataset from UCI repository...")
    german_credit = fetch_ucirepo(id=144)

    # Get features and targets
    X = german_credit.data.features
    y = german_credit.data.targets

    # Combine into one dataframe
    df = pd.concat([X, y], axis=1)

    print("✓ Successfully loaded complete dataset from UCI")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nTarget column name: {y.columns[0]}")
    print(f"Target values: {y[y.columns[0]].unique()}")

    # Save the complete dataset
    df.to_csv('complete_german_credit.csv', index=False)
    print("\n✓ Complete dataset saved to 'complete_german_credit.csv'")

except Exception as e:
    print(f"Could not load from UCI: {e}")
    print("\nTrying to check your local file structure...")

    # Check what files you have
    data_dir = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\German_Credit_Risk"
    files = os.listdir(data_dir)

    print(f"\nFiles in your directory:")
    for file in files:
        print(f"  - {file}")

    # Look for files that might contain the target
    csv_files = [f for f in files if f.endswith('.csv')]
    print(f"\nCSV files found: {csv_files}")

    # Load your existing file
    data_path = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\German_Credit_Risk\german_credit_data.csv"
    df = pd.read_csv(data_path)

    print(f"\nYour current file shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Check if any column might be the target
    print("\nChecking columns for potential target variable:")
    for col in df.columns:
        unique_vals = df[col].nunique()
        print(f"  {col}: {unique_vals} unique values")
        if unique_vals <= 3:  # Binary or small categorical might be target
            print(f"    Values: {df[col].unique()}")