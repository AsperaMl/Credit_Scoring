import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Бустинговый классификатор LightGBM
import lightgbm as lgb

# Для обработки категориальных переменных
from sklearn.preprocessing import LabelEncoder

# Для визуализации
import matplotlib.pyplot as plt
import seaborn as sns



filepath = "/datasets/application_train.csv"
df_train = pd.read_csv(filepath)



print("\nКоличество пропусков в каждом столбце:")
missing_values = df_train.isnull().sum().sort_values(ascending=False)
print(missing_values[missing_values > 0])


numeric_features = df_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df_train.select_dtypes(include=['object']).columns.tolist()
numeric_features.remove('TARGET')
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
print(df_train.head())

X = df_train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
y = df_train['TARGET']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер валидационной выборки: {X_val.shape}")

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)



params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

model_main = lgb.train(params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=10000,
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(50)])

import joblib

# Сохранение обученной модели
joblib.dump(model_main, 'joblib_models/lgbm_model.joblib')
print("Модель сохранена в файл 'lgbm_model.joblib'")
