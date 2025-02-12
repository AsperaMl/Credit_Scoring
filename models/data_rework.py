# обрабатываем трейн дату и вставляем в модель
import pandas as pd
import joblib
import sklearn as skl
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
import re
filepath = "/datasets/application_train.csv"
df_train = pd.read_csv(filepath)
df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
numeric_features = df_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df_train.select_dtypes(include=['object']).columns.tolist()
# numeric_features.remove('TARGET')
numeric_features.remove('SK_ID_CURR') if 'SK_ID_CURR' in numeric_features else None
categorical_features.remove('SK_ID_CURR') if 'SK_ID_CURR' in categorical_features else None

for col in numeric_features:
    df_train[col].fillna(df_train[col].median(), inplace=True)
for col in categorical_features:
    df_train[col].fillna(df_train[col].mode()[0], inplace=True)


print("\nОбщее количество оставшихся пропусков:", df_train.isnull().sum().sum())

le = LabelEncoder()
for col in categorical_features:
    df_train[col] = le.fit_transform(df_train[col])

print("\nДанные после кодирования категориальных признаков:")
print(df_train.head(20))
df_processed = df_train
df_processed.to_csv("application_train_prepared.csv")

