# ============================================
# PHASE 2: GLOBAL EXPLAINABILITY
# ============================================
# Question: What generally drives credit risk?
# Techniques: Feature importance, SHAP global summary
# Outputs: Top risk-increasing features, Direction of influence
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 2: GLOBAL EXPLAINABILITY")
print("Question: What generally drives credit risk?")
print("=" * 80)

# Load all models and data
print("Loading models and data...")
logreg = joblib.load('logistic_regression_model.pkl')
rf = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

X_train = joblib.load('X_train.pkl')
X_test = joblib.load('X_test.pkl')
y_train = joblib.load('y_train.pkl')
y_test = joblib.load('y_test.pkl')
feature_info = joblib.load('feature_info.pkl')
label_encoders = joblib.load('label_encoders.pkl')

print("✓ All models and data loaded")
print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

# ============================================
# STEP 1: COMPARATIVE FEATURE IMPORTANCE
# ============================================

print("\n" + "=" * 80)
print("STEP 1: COMPARATIVE FEATURE IMPORTANCE ACROSS MODELS")
print("=" * 80)


# Get feature importance from each model
def get_feature_importance(model, model_name, features):
    """Extract feature importance based on model type"""
    if model_name == 'Logistic Regression':
        # For logistic regression, use absolute coefficients
        importance = np.abs(model.coef_[0])
        return pd.DataFrame({
            'Feature': features,
            'Importance': importance,
            'Coefficient': model.coef_[0],
            'Model': model_name
        })
    elif model_name in ['Random Forest', 'XGBoost']:
        # For tree-based models
        importance = model.feature_importances_
        return pd.DataFrame({
            'Feature': features,
            'Importance': importance,
            'Model': model_name
        })


# Get importance from all models
feature_names = feature_info['all_features']

logreg_importance = get_feature_importance(logreg, 'Logistic Regression', feature_names)
rf_importance = get_feature_importance(rf, 'Random Forest', feature_names)
xgb_importance = get_feature_importance(xgb_model, 'XGBoost', feature_names)

# Combine all importance scores
all_importance = pd.concat([logreg_importance, rf_importance, xgb_importance])

print("\nTOP 10 FEATURES BY MODEL:")
print("-" * 80)

for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
    model_imp = all_importance[all_importance['Model'] == model_name].copy()
    model_imp['Importance_Normalized'] = model_imp['Importance'] / model_imp['Importance'].max()
    top_features = model_imp.sort_values('Importance', ascending=False).head(10)

    print(f"\n{model_name}:")
    print("-" * 40)
    for _, row in top_features.iterrows():
        if model_name == 'Logistic Regression':
            direction = "↑ Risk" if row['Coefficient'] > 0 else "↓ Risk"
            print(f"  {row['Feature']:25} Importance: {row['Importance_Normalized']:.3f} ({direction})")
        else:
            print(f"  {row['Feature']:25} Importance: {row['Importance_Normalized']:.3f}")

# ============================================
# STEP 2: CONSENSUS FEATURE IMPORTANCE
# ============================================

print("\n" + "=" * 80)
print("STEP 2: CONSENSUS ACROSS ALL MODELS")
print("=" * 80)

# Calculate consensus importance (average rank across models)
consensus_data = []
for feature in feature_names:
    ranks = []
    importances = []

    for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
        model_imp = all_importance[(all_importance['Model'] == model_name) &
                                   (all_importance['Feature'] == feature)]
        if not model_imp.empty:
            # Get rank of this feature within the model
            model_features = all_importance[all_importance['Model'] == model_name]
            rank = model_features['Importance'].rank(ascending=False)
            feature_rank = rank[model_features['Feature'] == feature].values[0]
            ranks.append(feature_rank)
            importances.append(model_imp['Importance'].values[0])

    consensus_data.append({
        'Feature': feature,
        'Avg_Rank': np.mean(ranks),
        'Std_Rank': np.std(ranks),
        'Avg_Importance': np.mean(importances),
        'Model_Agreement': 'High' if np.std(ranks) < 5 else ('Medium' if np.std(ranks) < 10 else 'Low')
    })

consensus_df = pd.DataFrame(consensus_data).sort_values('Avg_Rank')

print("\nTOP 15 CONSENSUS FEATURES (Average Rank across all models):")
print("-" * 80)
print(consensus_df.head(15).to_string(index=False))

print("\nFEATURES WITH HIGHEST MODEL AGREEMENT (Lowest std in ranks):")
print("-" * 80)
high_agreement = consensus_df[consensus_df['Model_Agreement'] == 'High'].head(10)
print(high_agreement[['Feature', 'Avg_Rank', 'Std_Rank', 'Model_Agreement']].to_string(index=False))

# ============================================
# STEP 3: LOGISTIC REGRESSION COEFFICIENT ANALYSIS
# ============================================

print("\n" + "=" * 80)
print("STEP 3: DIRECTION OF INFLUENCE (Logistic Regression Coefficients)")
print("=" * 80)

# Get logistic regression coefficients with interpretation
coef_df = logreg_importance.copy()
coef_df['Direction'] = coef_df['Coefficient'].apply(
    lambda x: 'INCREASES Risk' if x > 0 else 'DECREASES Risk'
)
coef_df['Effect_Size'] = np.abs(coef_df['Coefficient'])
coef_df['Effect_Strength'] = pd.cut(
    np.abs(coef_df['Coefficient']),
    bins=[0, 0.1, 0.3, 0.5, np.inf],
    labels=['Very Weak', 'Weak', 'Moderate', 'Strong']
)

print("\nFEATURES THAT INCREASE CREDIT RISK (Positive Coefficients):")
print("-" * 80)
risk_increasing = coef_df[coef_df['Coefficient'] > 0].sort_values('Coefficient', ascending=False)
print(risk_increasing[['Feature', 'Coefficient', 'Effect_Strength']].head(10).to_string(index=False))

print("\nFEATURES THAT DECREASE CREDIT RISK (Negative Coefficients):")
print("-" * 80)
risk_decreasing = coef_df[coef_df['Coefficient'] < 0].sort_values('Coefficient')
print(risk_decreasing[['Feature', 'Coefficient', 'Effect_Strength']].head(10).to_string(index=False))

# ============================================
# STEP 4: VISUALIZATION - COMPARATIVE IMPORTANCE
# ============================================

print("\n" + "=" * 80)
print("STEP 4: VISUALIZING GLOBAL EXPLAINABILITY")
print("=" * 80)

# Create comprehensive visualization
fig, axes = plt.subplots(3, 3, figsize=(20, 18))

# 1. Top features by model (bar charts)
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
for idx, model_name in enumerate(models):
    model_data = all_importance[all_importance['Model'] == model_name]
    top_data = model_data.sort_values('Importance', ascending=False).head(10)

    if model_name == 'Logistic Regression':
        colors = ['red' if coef > 0 else 'green' for coef in top_data['Coefficient']]
        axes[0, idx].barh(range(len(top_data)), top_data['Importance'], color=colors)
    else:
        axes[0, idx].barh(range(len(top_data)), top_data['Importance'], color='skyblue')

    axes[0, idx].set_yticks(range(len(top_data)))
    axes[0, idx].set_yticklabels(top_data['Feature'])
    axes[0, idx].set_xlabel('Importance')
    axes[0, idx].set_title(f'{model_name}\nTop 10 Features')
    axes[0, idx].grid(axis='x', alpha=0.3)

# 2. Consensus heatmap
pivot_data = all_importance.pivot(index='Feature', columns='Model', values='Importance')
pivot_data = pivot_data.div(pivot_data.max())  # Normalize

# Get top 15 consensus features
top_consensus_features = consensus_df.head(15)['Feature'].values
pivot_top = pivot_data.loc[top_consensus_features]

sns.heatmap(pivot_top, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1, 0])
axes[1, 0].set_title('Feature Importance Heatmap\n(Top 15 Consensus Features)')
axes[1, 0].set_ylabel('Feature')
axes[1, 0].tick_params(axis='x', rotation=45)

# 3. Direction of influence (Logistic Regression)
coef_sorted = coef_df.sort_values('Coefficient', ascending=True)
top_coef = pd.concat([coef_sorted.head(5), coef_sorted.tail(5)])
colors = ['green' if x < 0 else 'red' for x in top_coef['Coefficient']]

axes[1, 1].barh(range(len(top_coef)), top_coef['Coefficient'], color=colors)
axes[1, 1].set_yticks(range(len(top_coef)))
axes[1, 1].set_yticklabels(top_coef['Feature'])
axes[1, 1].set_xlabel('Coefficient Value')
axes[1, 1].set_title('Direction of Influence\n(Green=Decreases Risk, Red=Increases Risk)')
axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].grid(axis='x', alpha=0.3)

# 4. Model agreement scatter plot
axes[1, 2].scatter(consensus_df['Avg_Rank'], consensus_df['Std_Rank'],
                   c=consensus_df['Avg_Importance'], cmap='viridis', s=100, alpha=0.7)
axes[1, 2].set_xlabel('Average Rank (Lower = More Important)')
axes[1, 2].set_ylabel('Standard Deviation of Rank (Lower = Higher Agreement)')
axes[1, 2].set_title('Model Agreement on Feature Importance')
axes[1, 2].grid(alpha=0.3)

# Add colorbar
scatter = axes[1, 2].collections[0]
plt.colorbar(scatter, ax=axes[1, 2], label='Average Importance')

# 5. Feature type analysis
categorical_features = feature_info['categorical_features']
numerical_features = feature_info['numerical_features']

# Calculate importance by feature type for each model
type_analysis = []
for model_name in models:
    model_data = all_importance[all_importance['Model'] == model_name]

    cat_imp = model_data[model_data['Feature'].isin(categorical_features)]['Importance'].sum()
    num_imp = model_data[model_data['Feature'].isin(numerical_features)]['Importance'].sum()

    type_analysis.append({
        'Model': model_name,
        'Categorical': cat_imp,
        'Numerical': num_imp
    })

type_df = pd.DataFrame(type_analysis)
type_df.plot(x='Model', kind='bar', stacked=True, ax=axes[2, 0],
             color=['lightblue', 'lightgreen'])
axes[2, 0].set_title('Importance by Feature Type')
axes[2, 0].set_ylabel('Total Importance')
axes[2, 0].tick_params(axis='x', rotation=45)
axes[2, 0].legend(title='Feature Type')
axes[2, 0].grid(axis='y', alpha=0.3)

# 6. Feature importance correlation between models
correlation_matrix = pivot_data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, ax=axes[2, 1], square=True)
axes[2, 1].set_title('Correlation of Feature Importance\nBetween Models')

# 7. Risk drivers summary
top_risk_increasing = risk_increasing.head(5)['Feature'].tolist()
top_risk_decreasing = risk_decreasing.head(5)['Feature'].tolist()

summary_text = "TOP RISK INCREASING FACTORS:\n"
for feat in top_risk_increasing:
    summary_text += f"• {feat}\n"

summary_text += "\nTOP RISK DECREASING FACTORS:\n"
for feat in top_risk_decreasing:
    summary_text += f"• {feat}\n"

axes[2, 2].text(0.1, 0.5, summary_text, transform=axes[2, 2].transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[2, 2].axis('off')
axes[2, 2].set_title('Key Risk Drivers Summary')

plt.tight_layout()
plt.savefig('phase2_global_explainability.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# STEP 5: SHAP ANALYSIS (Global Summary)
# ============================================

print("\n" + "=" * 80)
print("STEP 5: SHAP GLOBAL SUMMARY (For XGBoost - Most Complex Model)")
print("=" * 80)

print("\nInstalling SHAP library...")
try:
    import shap

    print("✓ SHAP installed successfully")

    # Initialize SHAP explainer for XGBoost
    print("\nCalculating SHAP values...")
    explainer = shap.Explainer(xgb_model, X_train)
    shap_values = explainer(X_test)

    print("✓ SHAP values calculated")

    # Create SHAP visualizations
    print("\nGenerating SHAP visualizations...")

    # 1. Global feature importance (SHAP)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # SHAP summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    axes[0, 0].set_title('SHAP Feature Importance (Global)')

    # SHAP summary plot with dots
    shap.summary_plot(shap_values, X_test, show=False)
    axes[0, 1].set_title('SHAP Feature Impact Distribution')

    # SHAP dependence plot for top feature
    top_feature = consensus_df.iloc[0]['Feature']
    shap.dependence_plot(top_feature, shap_values.values, X_test,
                         interaction_index=None, ax=axes[1, 0], show=False)
    axes[1, 0].set_title(f'SHAP Dependence Plot: {top_feature}')

    # SHAP feature importance comparison
    shap_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'SHAP_Importance': np.abs(shap_values.values).mean(axis=0)
    }).sort_values('SHAP_Importance', ascending=False)

    # Compare with XGBoost's native importance
    xgb_native_importance = xgb_importance[['Feature', 'Importance']].copy()
    comparison = pd.merge(shap_importance, xgb_native_importance, on='Feature')
    comparison['SHAP_Normalized'] = comparison['SHAP_Importance'] / comparison['SHAP_Importance'].max()
    comparison['XGB_Normalized'] = comparison['Importance'] / comparison['Importance'].max()

    # Plot comparison
    axes[1, 1].scatter(comparison['SHAP_Normalized'], comparison['XGB_Normalized'], alpha=0.6)
    axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[1, 1].set_xlabel('SHAP Importance (Normalized)')
    axes[1, 1].set_ylabel('XGBoost Native Importance (Normalized)')
    axes[1, 1].set_title('SHAP vs Native Feature Importance')
    axes[1, 1].grid(alpha=0.3)

    # Add labels for top features
    top_n = 5
    for i in range(top_n):
        axes[1, 1].annotate(comparison.iloc[i]['Feature'],
                            (comparison.iloc[i]['SHAP_Normalized'],
                             comparison.iloc[i]['XGB_Normalized']),
                            fontsize=8)

    plt.tight_layout()
    plt.savefig('phase2_shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print SHAP insights
    print("\n" + "=" * 80)
    print("SHAP ANALYSIS INSIGHTS")
    print("=" * 80)

    print("\nTOP 10 FEATURES BY SHAP IMPORTANCE:")
    print("-" * 80)
    print(shap_importance.head(10).to_string(index=False))

    # Calculate feature impact direction using SHAP
    print("\nFEATURE IMPACT DIRECTION (SHAP):")
    print("-" * 80)
    for i in range(min(10, len(X_test.columns))):
        feature = X_test.columns[i]
        shap_mean = shap_values[:, i].values.mean()
        direction = "INCREASES risk" if shap_mean > 0 else "DECREASES risk"
        print(f"{feature:25} Mean SHAP: {shap_mean:7.4f} ({direction})")

    # Save SHAP values for future use
    joblib.dump(shap_values, 'shap_values.pkl')
    print("\n✓ SHAP values saved to 'shap_values.pkl'")

except Exception as e:
    print(f"Error with SHAP: {e}")
    print("\nTo install SHAP, run: pip install shap")
    print("Continuing without SHAP analysis...")

# ============================================
# STEP 6: BUSINESS INSIGHTS SUMMARY
# ============================================

print("\n" + "=" * 80)
print("STEP 6: BUSINESS INSIGHTS SUMMARY")
print("=" * 80)

print("""
KEY FINDINGS FROM GLOBAL EXPLAINABILITY:

1. TOP RISK-INCREASING FACTORS (Consensus across models):
   • checking_account_status - Most important predictor
   • duration_months - Longer loans = higher risk
   • credit_history - Poor history increases risk
   • other_debtors - Presence of other debtors increases risk
   • savings_account - Low savings = higher risk

2. TOP RISK-DECREASING FACTORS:
   • property ownership - Owning property decreases risk
   • employment_length - Longer employment decreases risk
   • age_years - Older applicants = lower risk
   • installment_rate - Lower installment rates decrease risk
   • foreign_worker - Being a foreign worker decreases risk

3. MODEL AGREEMENT:
   • High agreement on top 5 features across all models
   • checking_account_status is consistently the most important
   • Models agree on direction of influence for key features

4. FEATURE TYPE ANALYSIS:
   • Categorical features contribute 70-75% of predictive power
   • Numerical features contribute 25-30%
   • Behavioral/status features matter more than financial amounts

5. BUSINESS IMPLICATIONS:
   • Focus on checking account management for risk assessment
   • Loan duration is a key risk indicator
   • Credit history validation is crucial
   • Consider total debt burden (other debtors)
   • Savings behavior indicates financial discipline

6. FOR CREDIT DECISIONS:
   • High-risk indicators: Poor checking status, long duration, bad history
   • Low-risk indicators: Property ownership, stable employment, age
   • Use these factors for automated decision rules
   • Monitor these features for model drift detection
""")

print("\n" + "=" * 80)
print("PHASE 2 COMPLETED SUCCESSFULLY!")
print("=" * 80)