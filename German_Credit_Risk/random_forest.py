# ============================================
# PART 4: RANDOM FOREST - SEMI-BLACK BOX
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, auc)
from sklearn.model_selection import cross_val_score
import joblib

print("="*60)
print("RANDOM FOREST - SEMI-BLACK BOX MODEL")
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
print("TRAINING RANDOM FOREST MODEL")
print("="*60)

# Initialize Random Forest with optimized parameters
rf = RandomForestClassifier(
    n_estimators=100,           # Number of trees
    max_depth=10,               # Maximum depth of each tree
    min_samples_split=5,        # Minimum samples to split a node
    min_samples_leaf=2,         # Minimum samples in a leaf
    max_features='sqrt',        # Number of features to consider for split
    bootstrap=True,             # Use bootstrap samples
    random_state=42,            # Reproducibility
    class_weight='balanced_subsample',  # Handle class imbalance
    n_jobs=-1,                  # Use all processors
    verbose=0,
    oob_score=True              # Out-of-bag score for validation
)

print("Training Random Forest...")
rf.fit(X_train, y_train)

print("✓ Model trained successfully")
print(f"Number of trees: {rf.n_estimators}")
print(f"Out-of-bag score: {rf.oob_score_:.4f}")
print(f"Feature importance type: {rf.criterion}")

print("\n" + "="*60)
print("MODEL EVALUATION ON TEST SET")
print("="*60)

# Make predictions
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
roc_auc = roc_auc_score(y_test, y_pred_proba_rf)

print(f"\nPERFORMANCE METRICS:")
print("-"*40)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nCLASSIFICATION REPORT:")
print("-"*40)
print(classification_report(y_test, y_pred_rf,
                           target_names=['Good Credit', 'Bad Credit']))

print("\nCONFUSION MATRIX:")
print("-"*40)
cm = confusion_matrix(y_test, y_pred_rf)
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
cv_scores = cross_val_score(rf, X_train, y_train,
                           cv=5, scoring='roc_auc', n_jobs=-1)

print(f"5-Fold Cross-Validation ROC-AUC Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"\nMean ROC-AUC: {cv_scores.mean():.4f}")
print(f"Std ROC-AUC:  {cv_scores.std():.4f}")

print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_info['all_features'],
    'Importance': rf.feature_importances_,
    'Type': ['Categorical' if f in feature_info['categorical_features'] else 'Numerical'
             for f in feature_info['all_features']]
}).sort_values('Importance', ascending=False)

print("\nTOP 15 MOST IMPORTANT FEATURES (Gini Importance):")
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
axes[0, 0].barh(range(len(top_features)), top_features['Importance'])
axes[0, 0].set_yticks(range(len(top_features)))
axes[0, 0].set_yticklabels(top_features['Feature'])
axes[0, 0].set_xlabel('Gini Importance')
axes[0, 0].set_title('Top 15 Feature Importance')
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
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_rf)
roc_auc_value = auc(fpr, tpr)
axes[0, 2].plot(fpr, tpr, color='darkgreen', lw=2, label=f'Random Forest (AUC = {roc_auc_value:.3f})')
axes[0, 2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5, label='Random')
axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].set_title('ROC Curve')
axes[0, 2].legend(loc='lower right')
axes[0, 2].grid(alpha=0.3)

# 4. Confusion Matrix Heatmap
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# 5. Probability Distribution
axes[1, 1].hist(y_pred_proba_rf[y_test == 0], bins=30, alpha=0.5, label='Good Credit', color='green')
axes[1, 1].hist(y_pred_proba_rf[y_test == 1], bins=30, alpha=0.5, label='Bad Credit', color='red')
axes[1, 1].set_xlabel('Predicted Probability (Bad Credit)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Predicted Probability Distribution')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# 6. Cross-Validation Scores
axes[1, 2].plot(range(1, 6), cv_scores, 'o-', linewidth=2, markersize=8)
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
plt.savefig('random_forest_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("RANDOM FOREST TREE ANALYSIS")
print("="*60)

# Analyze individual trees
print(f"\nNumber of trees: {len(rf.estimators_)}")
print(f"Average tree depth: {np.mean([tree.get_depth() for tree in rf.estimators_]):.1f}")
print(f"Average leaf nodes per tree: {np.mean([tree.get_n_leaves() for tree in rf.estimators_]):.1f}")

# Sample tree structure
sample_tree = rf.estimators_[0]
print(f"\nSample Tree (Tree #0):")
print(f"  Depth: {sample_tree.get_depth()}")
print(f"  Number of leaves: {sample_tree.get_n_leaves()}")
print(f"  Number of nodes: {sample_tree.tree_.node_count}")

print("\n" + "="*60)
print("BUSINESS INTERPRETATION")
print("="*60)

print("""
KEY INSIGHTS FROM RANDOM FOREST:

1. ENSEMBLE ADVANTAGES:
   - Combines multiple decision trees to reduce overfitting
   - Provides feature importance based on Gini impurity reduction
   - Handles non-linear relationships better than Logistic Regression

2. SEMI-BLACK BOX NATURE:
   - Individual trees are interpretable
   - Ensemble makes overall model less interpretable
   - Feature importance provides global explanations

3. FEATURE INSIGHTS:
   - Gini importance shows which features reduce impurity most
   - Can identify interactions between features
   - Shows relative contribution of categorical vs numerical features

4. FOR PHASE 2 (XAI):
   - We'll use techniques like SHAP to explain individual predictions
   - Compare feature importance with Logistic Regression
   - Analyze decision paths for specific cases
""")

print("\n" + "="*60)
print("SAVING MODEL AND ARTIFACTS")
print("="*60)

# Save model and artifacts
joblib.dump(rf, 'random_forest_model.pkl')

# Save feature importance
feature_importance.to_csv('random_forest_feature_importance.csv', index=False)

# Save predictions for comparison
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_rf,
    'Probability_Bad': y_pred_proba_rf
})
predictions_df.to_csv('rf_predictions.csv', index=False)

print("✓ Random Forest model saved to 'random_forest_model.pkl'")
print("✓ Feature importance saved to 'random_forest_feature_importance.csv'")
print("✓ Predictions saved to 'rf_predictions.csv'")
print("✓ Visualizations saved to 'random_forest_results.png'")

print("\n" + "="*60)
print("PHASE 1 - PART 4 COMPLETED!")
print("="*60)
print("""
NEXT STEPS:
1. Run XGBoost (Black Box Model)
2. Compare all three models

Random Forest provides a balance between performance and interpretability.
The feature importance gives us insights into what drives credit decisions.
""")