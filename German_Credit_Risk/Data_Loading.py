# ============================================
# PART 1: SETUP AND DATA LOADING
# ============================================

# Install required packages (run this in terminal first)
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the dataset
data_path = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\German_Credit_Risk\german_credit_data.csv"
df = pd.read_csv(data_path)

print("="*50)
print("DATASET LOADED SUCCESSFULLY")
print("="*50)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

# Save for next parts
df.to_pickle('temp_data.pkl')