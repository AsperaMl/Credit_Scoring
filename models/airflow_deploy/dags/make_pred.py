import joblib
import pandas as pd

model = joblib.load('/opt/airflow/dags/final_lgbm_model_hyperopt.joblib_1')
filepath_test = "/opt/airflow/dags/test_processed_to_pred.csv"
df_test = pd.read_csv(filepath_test)
df_test.drop("Unnamed: 0", axis=1, inplace=True)
X_test = df_test.drop(['SK_ID_CURR'], axis=1)
predictions = model.predict_proba(X_test)[:, 1]
submission_filepath='subs_airflow.csv'
try:
    submission = pd.DataFrame({
        'SK_ID_CURR': df_test['SK_ID_CURR'],
        'TARGET': predictions
    })
    submission.to_csv(submission_filepath, index=False)
    print(f"Файл предсказаний сохранен как {submission_filepath}.")
except Exception as e:
    print(f"Ошибка при сохранении файла submission: {e}")
