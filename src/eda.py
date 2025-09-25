import pandas as pd

'''
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
labels_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
'''
data_url = "../secom/secom.data"
labels_url = "../secom/secom_labels.data"

sensor_data = pd.read_csv(data_url, sep = " ", header = None)
sensor_data.columns = [f"Sensor_{i+1}" for i in range(sensor_data.shape[1])]

labels = pd.read_csv(labels_url, sep = " ", header = None)
labels.columns = ["Status", "Timestamp"]

df_secom = pd.concat([sensor_data, labels], axis = 1)

def retrieve_data():
    return df_secom

def count_missing(df):
    missing_values = df.isnull().sum()
    return missing_values[missing_values > 0].sort_values(ascending=False)

print("Shape of the dataset (rows, columns):")
print("Sensor data: ", sensor_data.shape)
print("Label dat: ", labels.shape)
print(df_secom.shape)
print("-" * 40)

print("First 5 rows of the combined dataset:")
print(df_secom.head())
print("-" * 40)

print("Distribution of Pass/Fail Status:")
print(df_secom['Status'].value_counts())
print("-" * 40)

print("Missing value count per column, desc")
print(count_missing(df_secom))
print("-" * 40)

