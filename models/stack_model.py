import pandas as pd
import joblib
import sklearn as skl
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Загрузка данных
filepath = "application_train_prepared.csv"
df_train = pd.read_csv(filepath)
df_train.drop("Unnamed: 0", axis=1, inplace=True)
X = df_train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
y = df_train['TARGET']

# Разделение данных на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model_xgb = xgb.XGBClassifier(random_state=42, eval_metric='logloss',verbosity=1, n_jobs=-1)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1, n_jobs=-1)
filepath_model = "joblib_models/final_lgbm_model_hyperopt.joblib_1"
model_lgb = joblib.load(filepath_model)


print("Обучение XGBoost...")
model_xgb.fit(X_train, y_train)

print("Обучение Random Forest...")
model_rf.fit(X_train, y_train)


# Получение предсказаний базовых моделей
print("Получение предсказаний базовых моделей...")
train_pred_xgb = model_xgb.predict_proba(X_train)[:, 1]
val_pred_xgb = model_xgb.predict_proba(X_val)[:, 1]

train_pred_rf = model_rf.predict_proba(X_train)[:, 1]
val_pred_rf = model_rf.predict_proba(X_val)[:, 1]

train_pred_lgb = model_lgb.predict_proba(X_train)[:, 1]
val_pred_lgb = model_lgb.predict_proba(X_val)[:, 1]

# Создание новых признаков для финальной модели
print("Создание мета-признаков для логистической регрессии...")
X_train_meta = pd.DataFrame({
    'xgb_pred': train_pred_xgb,
    'rf_pred': train_pred_rf,
    'lgb_pred': train_pred_lgb
})

X_val_meta = pd.DataFrame({
    'xgb_pred': val_pred_xgb,
    'rf_pred': val_pred_rf,
    'lgb_pred': val_pred_lgb
})

# Обучение финальной модели (логистической регрессии)
print("Обучение финальной модели (логистическая регрессия)...")
final_model = LogisticRegression(random_state=42, max_iter=1000)
final_model.fit(X_train_meta, y_train)

# Оценка модели на валидационной выборке
print("Оценка модели...")
val_pred_final = final_model.predict_proba(X_val_meta)[:, 1]
auc_score = roc_auc_score(y_val, val_pred_final)
print(f"ROC AUC на валидационной выборке: {auc_score:.4f}")

# Сохранение финальной модели и базовых моделей (опционально)
print("Сохранение модели...")
stacking_model = {
    'model_xgb': model_xgb,
    'model_rf': model_rf,
    'model_lgb': model_lgb,
    'final_model': final_model
}
joblib.dump(stacking_model, 'joblib_models/stacking_model_logreg_final.joblib')

print("Готово!")
