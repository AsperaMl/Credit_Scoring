#FROM apache/airflow:2.6.0
#
## Копируем файл зависимостей в контейнер
#COPY requirements.txt /requirements.txt
#
## Переключаемся на пользователя airflow для установки зависимостей
#USER airflow
#
## Обновляем pip и устанавливаем зависимости
#RUN pip install --upgrade pip && pip install -r /requirements.txt
#
#
# Используем официальный образ Airflow в качестве базового
# Используем официальный образ Airflow в качестве базового
FROM apache/airflow:2.6.0

# Переключаемся на пользователя root для установки системных пакетов
USER root

# Обновляем список пакетов и устанавливаем libgomp1
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean

# Копируем файл зависимостей в контейнер
COPY requirements.txt /requirements.txt

# Переключаемся на пользователя airflow для установки Python-зависимостей
USER airflow

# Обновляем pip и устанавливаем зависимости из requirements.txt
RUN pip install --upgrade pip && pip install -r /requirements.txt
