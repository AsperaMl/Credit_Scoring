from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import lightgbm as lgb
import pandas as pd
import joblib
import sklearn as skl
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np


filepath = "last_processed_1.csv"
df_train = pd.read_csv(filepath)
df_train.drop("Unnamed: 0", axis=1, inplace=True)
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

model = lgb.train(params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=10000,
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(50)])

y_pred = model.predict(X_val, num_iteration=model.best_iteration)


auc = roc_auc_score(y_val, y_pred)
print(f"\nAUC ROC на валидационной выборке: {auc:.4f}")

print("roc-auc: ", auc)


