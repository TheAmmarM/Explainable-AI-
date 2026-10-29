# ============================================
# PART 2: PREPROCESSING FOR GERMAN CREDIT
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

print("="*50)
print("PROPER PREPROCESSING FOR GERMAN CREDIT DATASET")
print("="*50)

# Load the complete dataset
df = pd.read_csv('complete_german_credit.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3))

print("\n" + "="*50)
print("TARGET VARIABLE ANALYSIS")
print("="*50)

print(f"Target column: 'class'")
print(f"Target values: {df['class'].unique()}")
print(f"\nTarget distribution:")
print(df['class'].value_counts())
print(f"\nPercentage:")
print(df['class'].value_counts(normalize=True) * 100)

# Convert target: 1=good → 0, 2=bad → 1
df['Risk'] = df['class'].replace({1: 0, 2: 1})
df = df.drop('class', axis=1)

print(f"\nConverted target (0=good credit, 1=bad credit):")
print(df['Risk'].value_counts())
print(f"Good credit (0): {sum(df['Risk']==0)} samples")
print(f"Bad credit (1): {sum(df['Risk']==1)} samples")

print("\n" + "="*50)
print("DATA CLEANING AND ENCODING")
print("="*50)

# Check data types
print("\nData types:")
print(df.dtypes)

# The UCI German Credit dataset has all features as categorical/numeric codes
# Let's map them to proper feature names for interpretability
feature_names = {
    'Attribute1': 'checking_account_status',
    'Attribute2': 'duration_months',
    'Attribute3': 'credit_history',
    'Attribute4': 'purpose',
    'Attribute5': 'credit_amount',
    'Attribute6': 'savings_account',
    'Attribute7': 'employment_length',
    'Attribute8': 'installment_rate',
    'Attribute9': 'personal_status_sex',
    'Attribute10': 'other_debtors',
    'Attribute11': 'residence_since',
    'Attribute12': 'property',
    'Attribute13': 'age_years',
    'Attribute14': 'other_installment_plans',
    'Attribute15': 'housing',
    'Attribute16': 'existing_credits',
    'Attribute17': 'job',
    'Attribute18': 'dependents',
    'Attribute19': 'telephone',
    'Attribute20': 'foreign_worker'
}

# Rename columns for better understanding
df = df.rename(columns=feature_names)
print(f"\nRenamed columns for interpretability:")
print(df.columns.tolist())

# Separate features and target
X = df.drop('Risk', axis=1)
y = df['Risk']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Identify which columns are categorical vs numerical
# Based on German Credit dataset documentation:
categorical_features = [
    'checking_account_status', 'credit_history', 'purpose',
    'savings_account', 'employment_length', 'personal_status_sex',
    'other_debtors', 'property', 'other_installment_plans',
    'housing', 'job', 'telephone', 'foreign_worker'
]

numerical_features = [
    'duration_months', 'credit_amount', 'installment_rate',
    'residence_since', 'age_years', 'existing_credits', 'dependents'
]

print(f"\nCategorical features ({len(categorical_features)}):")
print(categorical_features)
print(f"\nNumerical features ({len(numerical_features)}):")
print(numerical_features)

# Check if all features are accounted for
all_features = categorical_features + numerical_features
missing = [col for col in X.columns if col not in all_features]
extra = [col for col in all_features if col not in X.columns]

if missing:
    print(f"\nWarning: Some features not categorized: {missing}")
if extra:
    print(f"\nWarning: Extra features in categorization: {extra}")

print("\n" + "="*50)
print("ENCODING CATEGORICAL VARIABLES")
print("="*50)

# Encode categorical variables (some might already be encoded as numbers)
label_encoders = {}
for col in categorical_features:
    if col in X.columns:
        # Check if already numeric
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded {col}: {len(le.classes_)} categories")
        else:
            print(f"{col} already numeric: {X[col].nunique()} unique values")

print("\n" + "="*50)
print("DATA SPLITTING")
print("="*50)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape} samples")
print(f"Test set: {X_test.shape} samples")
print(f"\nTraining target distribution: {np.bincount(y_train)}")
print(f"Test target distribution: {np.bincount(y_test)}")

print("\n" + "="*50)
print("SAVING ARTIFACTS")
print("="*50)

# Save all artifacts
joblib.dump(X_train, 'X_train.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Save feature information
feature_info = {
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'all_features': X.columns.tolist(),
    'target_name': 'Risk',
    'target_mapping': {0: 'Good Credit', 1: 'Bad Credit'}
}

joblib.dump(feature_info, 'feature_info.pkl')

# Save the cleaned dataframe
df.to_csv('cleaned_german_credit_interpretable.csv', index=False)

print("✓ X_train saved to 'X_train.pkl'")
print("✓ X_test saved to 'X_test.pkl'")
print("✓ y_train saved to 'y_train.pkl'")
print("✓ y_test saved to 'y_test.pkl'")
print("✓ Label encoders saved to 'label_encoders.pkl'")
print("✓ Feature info saved to 'feature_info.pkl'")
print("✓ Cleaned dataset saved to 'cleaned_german_credit_interpretable.csv'")

print("\n" + "="*50)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*50)

print(f"""
SUMMARY:
- Total samples: {len(df)}
- Features: {X.shape[1]} (7 numerical, 13 categorical)
- Target: 'Risk' (0=Good Credit, 1=Bad Credit)
- Class distribution: {sum(y==0)} good, {sum(y==1)} bad ({sum(y==1)/len(y)*100:.1f}% bad)
- Train/Test split: {len(X_train)}/{len(X_test)} samples

The dataset is now ready for modeling!
Features have been renamed for better interpretability.
""")

# Display sample of processed data
print("\nSample of processed features (first 3 rows):")
print(X.head(3))
print(f"\nCorresponding targets: {y[:3].values}")