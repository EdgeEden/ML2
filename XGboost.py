import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import time

# Load the dataset
df = pd.read_csv('synthetic_fraud_dataset.csv')

# Drop irrelevant columns
df = df.drop('Transaction_ID', axis=1)
df = df.drop('User_ID', axis=1)
# uncomment to drop Risk_Score
# df = df.drop('Risk_Score', axis=1)

# Convert Timestamp to datetime and extract features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df = df.drop('Timestamp', axis=1)

# Encode categorical variables
categorical_cols = ['Transaction_Type', 'Device_Type', 'Location',
                    'Merchant_Category', 'Card_Type', 'Authentication_Method']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df.drop('Fraud_Label', axis=1)
y = df['Fraud_Label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create models
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,  # Row subsampling
    colsample_bytree=0.8,  # Feature subsampling
    gamma=1,  # Regularization term
    random_state=42,  # random seed
    eval_metric='auc',  # Evaluation metric
    early_stopping_rounds=20,  # Stop training if no improvement
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train)  # Handle class imbalance
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,  # Maximum depth of the tree
    random_state=42,  # Random seed for reproducibility
    class_weight='balanced',  # Handle class imbalance
)

# Training
start_time = time.time()
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=2)  # Print evaluation every 2 rounds
training_time = time.time() - start_time

start_time = time.time()
rf_model.fit(X_train, y_train)
rf_training_time = time.time() - start_time

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

rf_y_pred = rf_model.predict(X_test)
rf_y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("\nXGBoost Classification Report:")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("---------------------------------------------------------------")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_y_pred))
print("\nRandom Forest ROC AUC Score:", roc_auc_score(y_test, rf_y_pred_proba))
print("---------------------------------------------------------------")

print(f"XGBoost Training Time: {training_time:.2f} seconds")
print(f"Random Forest Training Time: {rf_training_time:.2f} seconds")

# Feature importance
xgb.plot_importance(model, max_num_features=20)
plt.title('Feature Importance')
plt.show()
