import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, 
                           precision_recall_curve, 
                           average_precision_score,
                           confusion_matrix,
                           roc_auc_score)
import joblib
import warnings
from imblearn.under_sampling import RandomUnderSampler

# Configuration
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TEST_SIZE = 0.3
NOISE_SCALE = 0.15  # Noise level for synthetic data

# 1. Setup directories
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# 2. Data Preparation
def load_and_prepare_data():
    df = pd.read_csv("fraud_detection_dataset_updated.csv")
    
    # Feature engineering
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
    df.drop(['Time', 'Card_Number'], axis=1, inplace=True)
    
    # Add noise to synthetic data
    np.random.seed(RANDOM_STATE)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = numeric_cols.drop('Is_Fraud', errors='ignore')
    
    if len(numeric_cols) > 0:
        noise = NOISE_SCALE * np.random.randn(len(df), len(numeric_cols))
        df[numeric_cols] = df[numeric_cols] * (1 + noise)
    
    return df

# 3. Model Evaluation with Proper Metrics Handling
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Generate classification report as dictionary
    report_dict = classification_report(y_test, y_pred, 
                                      target_names=['Non-Fraud', 'Fraud'], 
                                      output_dict=True)
    
    # Extract metrics safely
    try:
        fraud_metrics = report_dict['Fraud']
        non_fraud_metrics = report_dict['Non-Fraud']
        weighted_avg = report_dict['weighted avg']
        
        cm = confusion_matrix(y_test, y_pred)
        ap_score = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Save full report
        with open(f'reports/{model_name}_report.txt', 'w') as f:
            f.write(f"{model_name.replace('_', ' ').title()} Evaluation Report\n")
            f.write("="*50 + "\n")
            f.write(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(cm, separator=', '))
            f.write(f"\n\nAverage Precision: {ap_score:.4f}")
            f.write(f"\nROC AUC Score: {roc_auc:.4f}")
        
        return {
            'fraud': {
                'precision': fraud_metrics['precision'],
                'recall': fraud_metrics['recall'],
                'f1': fraud_metrics['f1-score']
            },
            'non_fraud': {
                'precision': non_fraud_metrics['precision'],
                'recall': non_fraud_metrics['recall'],
                'f1': non_fraud_metrics['f1-score']
            },
            'weighted': {
                'precision': weighted_avg['precision'],
                'recall': weighted_avg['recall'],
                'f1': weighted_avg['f1-score']
            },
            'ap_score': ap_score,
            'roc_auc': roc_auc
        }
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        return None

# 4. Main Training Function
def train_and_evaluate():
    df = load_and_prepare_data()
    X = df.drop(columns=['Is_Fraud'])
    y = df['Is_Fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Balance classes (undersample majority)
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    
    # Get preprocessor
    preprocessor = get_preprocessor(X_train)
    
    # Define models with constrained parameters
    models = {
        'logistic_regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                solver='saga',
                random_state=RANDOM_STATE,
                C=0.1
            ))
        ]),
        'random_forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                class_weight='balanced_subsample',
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=RANDOM_STATE
            ))
        ]),
        'xgboost': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                scale_pos_weight=len(y[y==0])/len(y[y==1]),
                eval_metric='logloss',
                use_label_encoder=False,
                max_depth=3,
                n_estimators=100,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE
            ))
        ])
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name.replace('_', ' ').title()}...")
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, f'models/{name}_model.pkl')
        print(f"Model saved to models/{name}_model.pkl")
        
        # Evaluate
        eval_results = evaluate_model(model, X_test, y_test, name)
        if eval_results:
            results[name] = eval_results
            print(f"Evaluation report saved to reports/{name}_report.txt")
    
    return results

def get_preprocessor(X):
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

# 5. Run the training pipeline
if __name__ == "__main__":
    print("Starting fraud detection model training...")
    results = train_and_evaluate()
    
    # Print summary
    print("\nTraining Complete! Model Performance Summary:")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print("  Fraud Class:")
        print(f"    Precision: {metrics['fraud']['precision']:.4f}")
        print(f"    Recall:    {metrics['fraud']['recall']:.4f}")
        print(f"    F1-Score:  {metrics['fraud']['f1']:.4f}")
        print("\n  Non-Fraud Class:")
        print(f"    Precision: {metrics['non_fraud']['precision']:.4f}")
        print(f"    Recall:    {metrics['non_fraud']['recall']:.4f}")
        print(f"    F1-Score:  {metrics['non_fraud']['f1']:.4f}")
        print("\n  Weighted Avg:")
        print(f"    Precision: {metrics['weighted']['precision']:.4f}")
        print(f"    Recall:    {metrics['weighted']['recall']:.4f}")
        print(f"    F1-Score:  {metrics['weighted']['f1']:.4f}")
        print(f"\n  Avg Precision: {metrics['ap_score']:.4f}")
        print(f"  ROC AUC:      {metrics['roc_auc']:.4f}")