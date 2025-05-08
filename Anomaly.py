import pandas as pd
import numpy as np
from matplotlib.pyplot import subplot
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, \
    roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt

# Data Loading and Preprocessing
df = pd.read_csv('synthetic_fraud_dataset.csv')

# Drop Transaction_ID and User_ID
df = df.drop('Transaction_ID', axis=1)
df = df.drop('User_ID', axis=1)
df = df.drop('Risk_Score', axis=1)  # Uncomment to drop Risk_Score

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

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data - anomaly detection typically uses only normal data for training
# But we'll keep the test set with both classes for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Isolation Forest
iso_forest = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=0.3,  # auto estimates the contamination
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train, y_train)  # Fit on the entire training set

xgb = xgb.XGBClassifier(
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
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=2)


# Model Evaluation
iso_preds = iso_forest.predict(X_test)
iso_preds = np.where(iso_preds == 1, 0, 1)
scores = iso_forest.decision_function(X_test)
print(f"Isolation Forest Report")
print("Classification Report:")
print(classification_report(y_test, iso_preds, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, iso_preds, labels=[1, 0]).T)
iso_roc_auc = roc_auc_score(y_test, scores)
print(f"ROC AUC Score: {iso_roc_auc:.4f}")

xgb_pred = xgb.predict(X_test)
xgb_pred_proba = xgb.predict_proba(X_test)[:, 1]

print("---------------------------------------------------------------")
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_pred, labels=[1, 0]).T)
print("\nROC AUC Score:", roc_auc_score(y_test, xgb_pred_proba))


plt.figure(figsize=(12, 5))
# ROC Curves
subplot(1, 2, 1)
fpr, tpr, _ = roc_curve(y_test, scores)
roc_auc = roc_auc_score(y_test, scores)
plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')

# Precision-Recall Curves
plt.subplot(1, 2, 2)
precision, recall, _ = precision_recall_curve(y_test, scores)
pr_auc = auc(recall, precision)
plt.plot(recall, precision)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
