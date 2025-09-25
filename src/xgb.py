import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

data_url = "../secom/secom.data"
labels_url = "../secom/secom_labels.data"

sensor_data = pd.read_csv(data_url, sep=" ", header=None)
labels_data = pd.read_csv(labels_url, sep=" ", header=None)
labels_data.columns = ["Status", "Timestamp"]
df_secom = pd.concat([labels_data, sensor_data], axis=1)

X = df_secom.drop(columns=['Status', 'Timestamp'])
y = df_secom['Status']

# -1, 1 represented pass fail, switched to 1,0
y = y.replace({-1: 0, 1: 1})

missing_percentage = X.isnull().sum() / len(X)
threshold = 0.6
cols_to_drop = missing_percentage[missing_percentage > threshold].index
X_dropped = X.drop(columns=cols_to_drop)

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X_dropped), columns=X_dropped.columns)

selector = VarianceThreshold(threshold=0.0)
X_final = selector.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

ratio = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=ratio,      # ACCOUNT FOR CLASS IMBALANCE
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]


print("\n--- Model Evaluation Results ---")
print(classification_report(y_test, y_pred, target_names=['Pass (0)', 'Fail (1)']))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'n_estimators': [100, 200, 300, 400],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=100,  # try 100 different combinations
    scoring='roc_auc',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\n--- Hyperparameter Tuning Complete ---")
print(f"Best ROC AUC Score from search: {random_search.best_score_:.4f}")
print("Best parameters found:")
print(random_search.best_params_)

final_xgb_model = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=ratio,
    eval_metric='logloss',
    random_state=42,
    subsample=random_search.best_params_['subsample'],
    n_estimators=random_search.best_params_['n_estimators'],
    max_depth=random_search.best_params_['max_depth'],
    learning_rate=random_search.best_params_['learning_rate'],
    colsample_bytree=random_search.best_params_['colsample_bytree'],
)

print("\nTraining final model on the full training data...")
final_xgb_model.fit(X_train, y_train)

y_pred_final = final_xgb_model.predict(X_test)
y_pred_proba_final = final_xgb_model.predict_proba(X_test)[:, 1]

print("\n--- Final Model Performance on Test Set ---")
print(classification_report(y_test, y_pred_final, target_names=['Pass (0)', 'Fail (1)']))
print(f"Final ROC AUC Score on Test Set: {roc_auc_score(y_test, y_pred_proba_final):.4f}")
