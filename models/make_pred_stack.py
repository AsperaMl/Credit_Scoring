import joblib
import pandas as pd

# Загрузка стекинг-модели
model = joblib.load('joblib_models/stacking_model_logreg_final.joblib')
model_xgb = model['model_xgb']
model_rf = model['model_rf']
model_lgb = model['model_lgb']
final_model = model['final_model']

# Загрузка тестовых данных
df_test = pd.read_csv("airflow_deploy/dags/test_processed_to_pred.csv")
df_test.drop("Unnamed: 0", axis=1, inplace=True)
X_test = df_test.drop(['SK_ID_CURR'], axis=1)

# Получение предсказаний базовых моделей
xgb_pred = model_xgb.predict_proba(X_test)[:, 1]
rf_pred = model_rf.predict_proba(X_test)[:, 1]
lgb_pred = model_lgb.predict_proba(X_test)[:, 1]

# Создание мета-признаков
X_meta = pd.DataFrame({
    'xgb_pred': xgb_pred,
    'rf_pred': rf_pred,
    'lgb_pred': lgb_pred
})

# Получение окончательных предсказаний
predictions = final_model.predict_proba(X_meta)[:, 1]

# Сохранение предсказаний в CSV-файл
submission = pd.DataFrame({
    'SK_ID_CURR': df_test['SK_ID_CURR'],
    'TARGET': predictions
})

submission_filepath = 'submission_stack_1.csv'
try:
    submission.to_csv(submission_filepath, index=False)
    print(f"Файл предсказаний сохранен как '{submission_filepath}'.")
except Exception as e:
    print(f"Ошибка при сохранении файла submission: {e}")
