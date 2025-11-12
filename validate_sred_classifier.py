import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import numpy as np
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
try:
    import xgboost
except ImportError:
    print("Installing xgboost...")
    install('xgboost')
    import xgboost

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Installing scikit-learn...")
    install('scikit-learn')
    from sklearn.model_selection import train_test_split

# 1. Load and Clean Data
print("Loading and cleaning data...")
df = pd.read_csv('cleaned_training_dataset.csv')

if 'Governanace' in df.columns:
    df = df.drop(columns=['Governanace'])

df['Outage_Duration'].fillna(0, inplace=True)
df['Type_of_Submission'].fillna(df['Type_of_Submission'].mode()[0], inplace=True)
df.dropna(subset=['ITIL_Severity'], inplace=True)

# 2. Feature Engineering
print("Performing feature engineering...")
df['text_features'] = df['Headline_Description'].fillna('') + ' ' + df['Impact_Summary'].fillna('')

# 3. Define Features and Target
X = df.drop('SRED', axis=1)
y = df['SRED']

# Identify column types
categorical_features = [
    'Group', 'Organization', 'Closing_Code', 'Type_of_Submission', 
    'Activity', 'Day_Time_MW', 'Closing_Type', 'Prime_Element', 'ITIL_Severity'
]


numerical_features = ['Outage_Duration']
text_features_col = 'text_features'

# 4. Create Preprocessing Pipeline
print("Creating preprocessing pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(stop_words='english', max_features=1000), text_features_col),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='drop'
)

# 5. Create the XGBoost Pipeline
print("Creating the XGBoost pipeline...")
scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    ))
])

# 6. Perform K-Fold Cross-Validation
print("Performing 5-fold cross-validation...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['recall', 'precision', 'f1']

scores = cross_validate(xgb_pipeline, X, y, cv=kfold, scoring=scoring)

# 7. Report Results
print("\n--- Cross-Validation Results (5 Folds) ---")
print(f"Average Recall:    {scores['test_recall'].mean():.4f} (+/- {scores['test_recall'].std():.4f})")
print(f"Average Precision: {scores['test_precision'].mean():.4f} (+/- {scores['test_precision'].std():.4f})")
print(f"Average F1-score:  {scores['test_f1'].mean():.4f} (+/- {scores['test_f1'].std():.4f})")

print("\nScores for each fold:")
for i in range(5):
    print(f"  Fold {i+1}: Recall={scores['test_recall'][i]:.4f}, Precision={scores['test_precision'][i]:.4f}, F1={scores['test_f1'][i]:.4f}")

print("\nValidation complete.")
