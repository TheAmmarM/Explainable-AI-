# ============================================
# PART 5: XGBOOST MODEL - BLACK BOX MODEL
# ============================================

# ============================================
# train_xgboost.py - XGBOOST MODEL TRAINING
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, auc)
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("XGBOOST - BLACK BOX MODEL")
print("="*60)

# Import xgboost - use direct import without trying to rename
print("Importing XGBoost library...")
try:
    # This is the correct way to import
    from xgboost import XGBClassifier
    print("✓ XGBClassifier imported successfully")
except ImportError as e:
    print(f"Error: {e}")
    print("\nPossible solutions:")
    print("1. Make sure your file is NOT named 'xgboost.py' or 'xgboost_model.py'")
    print("2. Install xgboost with: pip install xgboost")
    print("3. Try restarting your Python environment")
    exit()

print("="*60)
print("XGBOOST - BLACK BOX MODEL (Most Complex)")
print("="*60)

# Load preprocessed data
X_train = joblib.load('X_train.pkl')
X_test = joblib.load('X_test.pkl')
y_train = joblib.load('y_train.pkl')
y_test = joblib.load('y_test.pkl')
feature_info = joblib.load('feature_info.pkl')

print("\n✓ Data loaded successfully")
print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Classes: Good Credit (0), Bad Credit (1)")

print("\n" + "="*60)
print("DATA PREPARATION FOR XGBOOST")
print("="*60)

# Calculate scale_pos_weight for handling class imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"Class distribution - Good: {sum(y_train == 0)}, Bad: {sum(y_train == 1)}")
print(f"scale_pos_weight: {scale_pos_weight:.2f} (automatically handles imbalance)")

print("\n" + "="*60)
print("TRAINING XGBOOST MODEL")
print("="*60)

# Define XGBoost parameters
params = {
    'objective': 'binary:logistic',     # Binary classification
    'eval_metric': 'logloss',           # Evaluation metric
    'booster': 'gbtree',                # Tree-based model
    'max_depth': 5,                     # Maximum tree depth
    'learning_rate': 0.1,               # Step size shrinkage
    'n_estimators': 100,                # Number of trees
    'subsample': 0.8,                   # Subsample ratio of training data
    'colsample_bytree': 0.8,            # Subsample ratio of columns
    'min_child_weight': 1,              # Minimum sum of instance weight
    'gamma': 0,                         # Minimum loss reduction for split
    'reg_alpha': 0.1,                   # L1 regularization
    'reg_lambda': 1.0,                  # L2 regularization
    'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
    'random_state': 42,                 # Reproducibility
    'n_jobs': -1,                       # Use all processors
    'verbosity': 0,
    'use_label_encoder': False
}

print("XGBoost Parameters:")
for key, value in params.items():
    print(f"  {key}: {value}")

# Initialize and train XGBoost
xgb_model = XGBClassifier(**params)
print("\nTraining XGBoost model...")
xgb_model.fit(X_train, y_train)

print("✓ Model trained successfully")
print(f"Number of boosting rounds: {xgb_model.n_estimators}")

print("\n" + "="*60)
print("MODEL EVALUATION ON TEST SET")
print("="*60)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_xgb)
precision = precision_score(y_test, y_pred_xgb)
recall = recall_score(y_test, y_pred_xgb)
f1 = f1_score(y_test, y_pred_xgb)
roc_auc = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"\nPERFORMANCE METRICS:")
print("-"*40)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nCLASSIFICATION REPORT:")
print("-"*40)
print(classification_report(y_test, y_pred_xgb,
                           target_names=['Good Credit', 'Bad Credit']))

print("\nCONFUSION MATRIX:")
print("-"*40)
cm = confusion_matrix(y_test, y_pred_xgb)
cm_df = pd.DataFrame(cm,
                     index=['Actual Good', 'Actual Bad'],
                     columns=['Predicted Good', 'Predicted Bad'])
print(cm_df)

# Calculate additional metrics
TN, FP, FN, TP = cm.ravel()
print(f"\nDetailed Analysis:")
print(f"True Negatives (Good→Good): {TN}")
print(f"False Positives (Good→Bad): {FP}")
print(f"False Negatives (Bad→Good): {FN}")
print(f"True Positives (Bad→Bad): {TP}")
print(f"\nError Rate: {(FP+FN)/len(y_test):.4f}")
print(f"Sensitivity (Recall): {TP/(TP+FN):.4f}")
print(f"Specificity: {TN/(TN+FP):.4f}")

print("\n" + "="*60)
print("CROSS-VALIDATION PERFORMANCE")
print("="*60)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(xgb_model, X_train, y_train,
                           cv=5, scoring='roc_auc', n_jobs=-1)

print(f"5-Fold Cross-Validation ROC-AUC Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"\nMean ROC-AUC: {cv_scores.mean():.4f}")
print(f"Std ROC-AUC:  {cv_scores.std():.4f}")

print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get feature importance (default is 'weight')
importance_scores = xgb_model.feature_importances_
feature_names = feature_info['all_features']

# Create feature importance dataframe
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance_scores,
    'Type': ['Categorical' if f in feature_info['categorical_features'] else 'Numerical'
             for f in feature_names]
}).sort_values('Importance', ascending=False)

print("\nTOP 15 MOST IMPORTANT FEATURES (Weight Importance):")
print("-"*60)
print(feature_importance.head(15).to_string(index=False))

print("\nFEATURE IMPORTANCE BY TYPE:")
print("-"*40)
categorical_imp = feature_importance[feature_importance['Type'] == 'Categorical']['Importance'].sum()
numerical_imp = feature_importance[feature_importance['Type'] == 'Numerical']['Importance'].sum()
print(f"Categorical features: {categorical_imp:.4f} ({categorical_imp*100:.1f}%)")
print(f"Numerical features:   {numerical_imp:.4f} ({numerical_imp*100:.1f}%)")

print("\nTOP 5 CATEGORICAL FEATURES:")
print("-"*40)
cat_top = feature_importance[feature_importance['Type'] == 'Categorical'].head(5)
print(cat_top[['Feature', 'Importance']].to_string(index=False))

print("\nTOP 5 NUMERICAL FEATURES:")
print("-"*40)
num_top = feature_importance[feature_importance['Type'] == 'Numerical'].head(5)
print(num_top[['Feature', 'Importance']].to_string(index=False))

print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Feature Importance Bar Plot
top_features = feature_importance.head(15)
axes[0, 0].barh(range(len(top_features)), top_features['Importance'], color='darkred')
axes[0, 0].set_yticks(range(len(top_features)))
axes[0, 0].set_yticklabels(top_features['Feature'])
axes[0, 0].set_xlabel('Importance (Weight)')
axes[0, 0].set_title('Top 15 Feature Importance\n(Frequency in trees)')
axes[0, 0].grid(axis='x', alpha=0.3)

# 2. Feature Type Importance
feature_types = ['Categorical', 'Numerical']
type_importance = [categorical_imp, numerical_imp]
colors = ['lightblue', 'lightgreen']
axes[0, 1].bar(feature_types, type_importance, color=colors)
axes[0, 1].set_ylabel('Total Importance')
axes[0, 1].set_title('Feature Importance by Type')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(type_importance):
    axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')

# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_xgb)
roc_auc_value = auc(fpr, tpr)
axes[0, 2].plot(fpr, tpr, color='darkred', lw=2, label=f'XGBoost (AUC = {roc_auc_value:.3f})')
axes[0, 2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5, label='Random')
axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].set_title('ROC Curve')
axes[0, 2].legend(loc='lower right')
axes[0, 2].grid(alpha=0.3)

# 4. Confusion Matrix Heatmap
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# 5. Probability Distribution
axes[1, 1].hist(y_pred_proba_xgb[y_test == 0], bins=30, alpha=0.5, label='Good Credit', color='green')
axes[1, 1].hist(y_pred_proba_xgb[y_test == 1], bins=30, alpha=0.5, label='Bad Credit', color='red')
axes[1, 1].set_xlabel('Predicted Probability (Bad Credit)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Predicted Probability Distribution')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# 6. Cross-Validation Scores
axes[1, 2].plot(range(1, 6), cv_scores, 'o-', linewidth=2, markersize=8, color='darkred')
axes[1, 2].axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
axes[1, 2].fill_between(range(1, 6),
                        cv_scores.mean() - cv_scores.std(),
                        cv_scores.mean() + cv_scores.std(),
                        alpha=0.2, color='gray')
axes[1, 2].set_xlabel('Fold Number')
axes[1, 2].set_ylabel('ROC-AUC Score')
axes[1, 2].set_title('5-Fold Cross-Validation Performance')
axes[1, 2].set_xticks(range(1, 6))
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('xgboost_model_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("BLACK BOX CHARACTERISTICS ANALYSIS")
print("="*60)

print("""
WHY XGBOOST IS A "BLACK BOX":

1. COMPLEXITY:
   - Gradient boosting with sequential tree building
   - Complex feature interactions across multiple trees
   - Non-linear transformations and splits

2. HARD TO INTERPRET:
   - Individual trees can be deep and complex
   - Final prediction combines hundreds of trees
   - Feature importance is aggregated, not per-instance

3. WHAT WE CAN STILL UNDERSTAND:
   - Global feature importance (weight, gain, cover)
   - Direction of feature impact (through SHAP in Phase 2)
   - Model behavior patterns

4. PERFORMANCE TRADEOFF:
   - Typically highest predictive accuracy
   - Most difficult to explain to stakeholders
   - Requires XAI techniques for explanations
""")

print("\n" + "="*60)
print("SAVING MODEL AND ARTIFACTS")
print("="*60)

# Save model and artifacts
joblib.dump(xgb_model, 'xgboost_model.pkl')

# Save feature importance
feature_importance.to_csv('xgboost_feature_importance.csv', index=False)

# Save predictions for comparison
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_xgb,
    'Probability_Bad': y_pred_proba_xgb
})
predictions_df.to_csv('xgboost_predictions.csv', index=False)

print("✓ XGBoost model saved to 'xgboost_model.pkl'")
print("✓ Feature importance saved to 'xgboost_feature_importance.csv'")
print("✓ Predictions saved to 'xgboost_predictions.csv'")
print("✓ Visualizations saved to 'xgboost_model_results.png'")

print("\n" + "="*60)
print("PHASE 1 - PART 5 COMPLETED!")
print("="*60)