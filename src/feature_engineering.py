import pandas as pd
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