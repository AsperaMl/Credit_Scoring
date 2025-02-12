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
filepath = "last_processed_2.csv"
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
# Определение пространства поиска гиперпараметров
space = {
    'num_leaves': scope.int(hp.quniform('num_leaves', 20, 150, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 50)),
    'min_child_samples': scope.int(hp.quniform('min_child_samples', 10, 100, 5)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'max_depth': scope.int(hp.quniform('max_depth', 5, 50, 1)),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-3), np.log(10)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-3), np.log(10)),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'goss']),
    'objective': 'binary',
    'metric': 'auc',
    'verbose': -1,
    'random_state': 42
}


def objective(params):
    # Создание объекта модели с заданными гиперпараметрами
    model = lgb.LGBMClassifier(
        num_leaves=params['num_leaves'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        min_child_samples=params['min_child_samples'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        max_depth=params['max_depth'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        boosting_type=params['boosting_type'],
        objective='binary',
        metric='auc',
        verbose=-1,
        random_state=42
    )

    # Обучение модели
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(50)]
    )

    # Предсказания вероятностей на валидационной выборке
    y_pred = model.predict_proba(X_val)[:, 1]

    # Вычисление AUC ROC
    auc = roc_auc_score(y_val, y_pred)

    # Hyperopt минимизирует функцию, поэтому возвращаем отрицательное значение
    return {'loss': -auc, 'status': STATUS_OK}
# Инициализация объекта Trials для хранения результатов
trials = Trials()

# Запуск оптимизации
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10,  # Количество итераций поиска
    trials=trials,
    rstate=np.random.default_rng(42)
)

print("\nЛучшие гиперпараметры:")
print(best)
boosting_types = ['gbdt', 'dart', 'goss']
best['boosting_type'] = boosting_types[best['boosting_type']]
# Объединение обучающей и валидационной выборки
X_full_train = pd.concat([X_train, X_val], axis=0)
y_full_train = pd.concat([y_train, y_val], axis=0)

# Создание объекта модели с лучшими гиперпараметрами
final_model = lgb.LGBMClassifier(
    num_leaves=int(best['num_leaves']),
    learning_rate=best['learning_rate'],
    n_estimators=int(best['n_estimators']),
    min_child_samples=int(best['min_child_samples']),
    subsample=best['subsample'],
    colsample_bytree=best['colsample_bytree'],
    max_depth=int(best['max_depth']),
    reg_alpha=best['reg_alpha'],
    reg_lambda=best['reg_lambda'],
    boosting_type=best['boosting_type'],
    objective='binary',
    metric='auc',
    verbose=-1,
    random_state=42,
)

# Обучение финальной модели
final_model.fit(
    X_full_train, y_full_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(50)]
)

# Сохранение модели
joblib.dump(final_model, 'joblib_models/final_lgbm_model_hyperopt.joblib_1')
print("\nФинальная модель сохранена как 'final_lgbm_model_hyperopt.joblib'")
# Предсказания вероятностей на валидационной выборке
y_final_pred = final_model.predict_proba(X_val)[:, 1]

# Вычисление AUC ROC
final_auc = roc_auc_score(y_val, y_final_pred)
print(f"\nФинальное AUC ROC на валидационной выборке: {final_auc:.4f}")
