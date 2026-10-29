# ============================================
# PART 3: LOGISTIC REGRESSION - INTERPRETABLE BASELINE
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
import joblib

print("="*60)
print("LOGISTIC REGRESSION - INTERPRETABLE BASELINE MODEL")
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
print(f"Features: {len(feature_info['all_features'])}")
print(f"Target: {feature_info['target_name']}")

# Scale numerical features for Logistic Regression
print("\n" + "="*60)
print("SCALING NUMERICAL FEATURES")
print("="*60)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Scale only numerical features
numerical_features = feature_info['numerical_features']
X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

print(f"Scaled {len(numerical_features)} numerical features:")
for feature in numerical_features:
    print(f"  - {feature}")

print("\n" + "="*60)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("="*60)

# Initialize and train Logistic Regression
logreg = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    solver='lbfgs',
    C=1.0,
    penalty='l2'  # Ridge regularization
)

logreg.fit(X_train_scaled, y_train)

print("✓ Model trained successfully")
print(f"Number of iterations: {logreg.n_iter_[0]}")
print(f"Classes: {logreg.classes_}")

print("\n" + "="*60)
print("MODEL EVALUATION ON TEST SET")
print("="*60)

# Make predictions
y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nPERFORMANCE METRICS:")
print("-"*40)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f} (Ability to identify bad credit correctly)")
print(f"Recall:    {recall:.4f} (Ability to find all bad credit cases)")
print(f"F1-Score:  {f1:.4f} (Balance of Precision and Recall)")
print(f"ROC-AUC:   {roc_auc:.4f} (Overall discriminative power)")

print("\nCLASSIFICATION REPORT:")
print("-"*40)
print(classification_report(y_test, y_pred,
                           target_names=['Good Credit', 'Bad Credit']))

print("\nCONFUSION MATRIX:")
print("-"*40)
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index=['Actual Good', 'Actual Bad'],
                     columns=['Predicted Good', 'Predicted Bad'])
print(cm_df)

# Calculate additional metrics from confusion matrix
TN, FP, FN, TP = cm.ravel()
print(f"\nDetailed Analysis:")
print(f"True Negatives (Good→Good): {TN}")
print(f"False Positives (Good→Bad): {FP} (Type I Error)")
print(f"False Negatives (Bad→Good): {FN} (Type II Error)")
print(f"True Positives (Bad→Bad): {TP}")
print(f"\nError Rate: {(FP+FN)/len(y_test):.4f}")
print(f"Specificity (True Negative Rate): {TN/(TN+FP):.4f}")

print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get coefficients
feature_importance = pd.DataFrame({
    'Feature': feature_info['all_features'],
    'Coefficient': logreg.coef_[0],
    'Abs_Coefficient': np.abs(logreg.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTOP 10 MOST IMPORTANT FEATURES:")
print("-"*40)
print(feature_importance.head(10).to_string(index=False))

print("\nTOP 5 FEATURES INCREASING BAD CREDIT RISK (Positive Coefficients):")
print("-"*40)
risk_increasing = feature_importance[feature_importance['Coefficient'] > 0].head(5)
print(risk_increasing[['Feature', 'Coefficient']].to_string(index=False))

print("\nTOP 5 FEATURES DECREASING BAD CREDIT RISK (Negative Coefficients):")
print("-"*40)
risk_decreasing = feature_importance[feature_importance['Coefficient'] < 0].head(5)
print(risk_decreasing[['Feature', 'Coefficient']].to_string(index=False))

print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Feature Coefficients
top_features = feature_importance.head(15).sort_values('Coefficient', ascending=True)
colors = ['red' if x > 0 else 'green' for x in top_features['Coefficient']]
axes[0, 0].barh(range(len(top_features)), top_features['Coefficient'], color=colors)
axes[0, 0].set_yticks(range(len(top_features)))
axes[0, 0].set_yticklabels(top_features['Feature'])
axes[0, 0].set_xlabel('Coefficient Value')
axes[0, 0].set_title('Feature Coefficients\n(Red = Bad Credit Risk, Green = Good Credit Risk)')
axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
axes[0, 0].grid(axis='x', alpha=0.3)

# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'LogReg (AUC = {roc_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5, label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend(loc='lower right')
axes[0, 1].grid(alpha=0.3)

# 3. Confusion Matrix Heatmap
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# 4. Probability Distribution
axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, label='Good Credit', color='green')
axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.5, label='Bad Credit', color='red')
axes[1, 1].set_xlabel('Predicted Probability (Bad Credit)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Predicted Probability Distribution')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_regression_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("BUSINESS INTERPRETATION")
print("="*60)

print("""
KEY INSIGHTS FROM LOGISTIC REGRESSION:

1. MODEL PERFORMANCE:
   - The model achieves reasonable accuracy on the test set
   - ROC-AUC shows good discriminative power (>0.75 is good)
   - Precision-Recall tradeoff indicates model behavior

2. FEATURE INTERPRETATION:
   - POSITIVE coefficients increase probability of BAD credit
   - NEGATIVE coefficients increase probability of GOOD credit
   - Magnitude shows relative importance

3. BUSINESS IMPLICATIONS:
   - False Positives: Good customers rejected (lost revenue)
   - False Negatives: Bad customers approved (potential losses)
   - The cost of each error type differs in credit risk

4. FOR PHASE 2 (XAI):
   - These coefficients provide baseline feature importance
   - We'll compare with more complex models
   - Individual predictions can be explained using these coefficients
""")

print("\n" + "="*60)
print("SAVING MODEL AND ARTIFACTS")
print("="*60)

# Save model and artifacts
joblib.dump(logreg, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'logreg_scaler.pkl')

# Save predictions for comparison
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Probability_Bad': y_pred_proba
})
predictions_df.to_csv('logreg_predictions.csv', index=False)

print("✓ Logistic Regression model saved to 'logistic_regression_model.pkl'")
print("✓ Scaler saved to 'logreg_scaler.pkl'")
print("✓ Predictions saved to 'logreg_predictions.csv'")
print("✓ Visualizations saved to 'logistic_regression_results.png'")

print("\n" + "="*60)
print("PHASE 1 - PART 3 COMPLETED!")
print("="*60)
print("""
NEXT STEPS:
1. Run Random Forest (Semi-Black Box)
2. Run XGBoost (Black Box)
3. Compare all three models

Logistic Regression serves as our interpretable baseline.
We can now explain predictions using feature coefficients!
""")