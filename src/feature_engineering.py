import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from src.eda import retrieve_data, count_missing

df = retrieve_data()

# Dropping sensors with over 60% missing values:
X = df.drop(columns = ["Status", "Timestamp"])
y = df["Status"]

missing_percentage = X.isnull().sum() / len(X)
threshold = 0.60

columns_to_drop = missing_percentage[missing_percentage > threshold].index

X_dropped = X.drop(columns = columns_to_drop)

print(f"Original number of features: {X.shape[1]}")
print(f"Number of features dropped: {len(columns_to_drop)}")
print(f"Number of features remaining: {X_dropped.shape[1]}")
print(count_missing(X_dropped))

imputer = SimpleImputer(strategy="median", fill_value=-np.inf)
X_imputed = imputer.fit_transform(X_dropped)
X_imputed = pd.DataFrame(X_imputed, columns=X_dropped.columns)
print(X_imputed.head())

# removal of low var features
selector = VarianceThreshold(threshold=0)
X_reduced = selector.fit_transform(X_imputed)

print(f"Original number of features: {X_imputed.shape[1]}")
print(f"Features remaining after removing zero-variance features: {X_reduced.shape[1]}")

def get_data():
    return X_reduced, y