import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import numpy as np



# 1. Load and Clean Data
print("Loading and cleaning data...")
df = pd.read_csv('cleaned_training_dataset.csv')

# Handle nulls
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

# 5. Train-Test Split
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Model Training
print("Training the XGBoost classifier...")
# Calculate scale_pos_weight for handling class imbalance
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# Create the XGBoost pipeline
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

# Train the model
xgb_pipeline.fit(X_train, y_train)

# 7. Evaluation
print("Evaluating the model...")
y_pred = xgb_pipeline.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print("                 Predicted")
print("                 0       1")
print("Actual    0 | {:<6}  {:<6}".format(confusion_matrix(y_test, y_pred)[0][0], confusion_matrix(y_test, y_pred)[0][1]))
print("          1 | {:<6}  {:<6}".format(confusion_matrix(y_test, y_pred)[1][0], confusion_matrix(y_test, y_pred)[1][1]))
print("\n")

# Feature Importance
try:
    feature_names = xgb_pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = xgb_pipeline.named_steps['classifier'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n--- Top 20 Feature Importances ---")
    print(feature_importance_df.head(20))
except Exception as e:
    print(f"\nCould not display feature importances: {e}")

print("\nTraining and evaluation complete.")
