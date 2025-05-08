import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

def load_and_preprocess():
    # 1. 数据加载与预处理
    data = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

    # 删除非数值列和常量列
    feature = data.drop('Flow ID', axis=1)
    feature = feature.drop(' Source IP', axis=1)
    feature = feature.drop(' Destination IP', axis=1)
    feature = feature.drop(' Timestamp', axis=1)

    # 编码标签
    le = LabelEncoder()
    data[' Label'] = le.fit_transform(data[' Label'])  # BENIGN=0, DDoS=1
    Y = data[' Label']
    features = feature.drop([' Label'], axis=1)

    # 处理无限大值和缺失值
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)

    # 数据标准化
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, Y, test_size=0.3, random_state=42, stratify=Y)
    return X_train, X_test, y_train, y_test, scaled_features, Y


def XGBoost(X_train, X_test, y_train, y_test):
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    print("\nXGBoost:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")


# 隔离森林模型
def iso_forest(X_scaled, Y):
    X_normal = X_scaled[Y == 0]
    iso_forest = IsolationForest(n_estimators=150, max_samples=256,
                                 contamination=0.44, random_state=42, n_jobs=-1)
    iso_forest.fit(X_normal)
    anomaly_scores = iso_forest.decision_function(X_scaled)
    anomalies = np.where(iso_forest.predict(X_scaled) == 1, 0, 1)

    print("\nIsolation Forest:")
    print(classification_report(Y, anomalies))
    print(f"ROC AUC: {roc_auc_score(Y, anomaly_scores):.4f}")


def OCSVM(X_scaled, Y):
    X_normal = X_scaled[Y == 0]
    oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
    oc_svm.fit(X_normal)
    anomaly_scores = oc_svm.decision_function(X_scaled)
    anomalies = np.where(oc_svm.predict(X_scaled) == 1, 0, 1)

    print("\nOne-Class SVM:")
    print(classification_report(y, anomalies))
    print(f"ROC AUC: {roc_auc_score(y, anomaly_scores):.4f}")


def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X_scaled, y = load_and_preprocess()
    XGBoost(X_train, X_test, y_train, y_test)
    iso_forest(X_scaled, y)
    OCSVM(X_scaled, y)
    # 可视化随机森林的特征重要性
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    feature_names = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv').drop(
        ['Flow ID', ' Source IP', ' Destination IP', ' Timestamp', ' Label'], axis=1).columns
    plot_feature_importance(rf, feature_names, "Random Forest Feature Importance")
