"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st


def evaluate_input(unique_data_path: str, endpoint: object) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param unique_data_path: путь до уникальных значений
    :param endpoint: endpoint
    """
    with open(unique_data_path) as file:
        unique_df = json.load(file)

    # поля для вводы данных, используем уникальные значения
    revolving = st.sidebar.slider(
        "Revolving", min_value=min(unique_df["Revolving"]), max_value=max(unique_df["Revolving"])
    )
    age = st.sidebar.slider(
        "age", min_value=min(unique_df["age"]), max_value=max(unique_df["age"])
    )
    numberoftime30 = st.sidebar.slider(
        "NumberOfTime30", min_value=min(unique_df["NumberOfTime30"]), max_value=max(unique_df["NumberOfTime30"])
    )
    debtratio = st.sidebar.slider(
        "DebtRatio", min_value=min(unique_df["DebtRatio"]), max_value=max(unique_df["DebtRatio"])
    )
    monthlyincome = st.sidebar.number_input(
        "MonthlyIncome",
        min_value=min(unique_df["MonthlyIncome"]),
        max_value=max(unique_df["MonthlyIncome"]),
    )
    numberofopen = st.sidebar.slider(
        "NumberOfOpen", min_value=min(unique_df["NumberOfOpen"]), max_value=max(unique_df["NumberOfOpen"])
    )
    numberoftimes90 = st.sidebar.slider(
        "NumberOfTimes90", min_value=min(unique_df["NumberOfTimes90"]), max_value=max(unique_df["NumberOfTimes90"])
    )
    numberrealestate = st.sidebar.slider(
        "NumberRealEstate", min_value=min(unique_df["NumberRealEstate"]), max_value=max(unique_df["NumberRealEstate"])
    )
    numberoftime60 = st.sidebar.slider(
        "NumberOfTime60", min_value=min(unique_df["NumberOfTime60"]), max_value=max(unique_df["NumberOfTime60"])
    )
    numberofdependents = st.sidebar.slider(
        "NumberOfDependents", min_value=min(unique_df["NumberOfDependents"]), max_value=max(unique_df["NumberOfDependents"])
    )

    dict_data = {
        "Revolving": revolving,
        "age": age,
        "NumberOfTime30": numberoftime30,
        "DebtRatio": debtratio,
        "MonthlyIncome": monthlyincome,
        "NumberOfOpen": numberofopen,
        "NumberOfTimes90": numberoftimes90,
        "NumberRealEstate": numberrealestate,
        "NumberOfTime60": numberoftime60,
        "NumberOfDependents": numberofdependents,
    }

    st.write(
        f"""### Данные клиента:\n
    1) Revolving: {dict_data['Revolving']}
    2) age: {dict_data['age']}
    3) NumberOfTime30: {dict_data['NumberOfTime30']}
    4) DebtRatio: {dict_data['DebtRatio']}
    5) MonthlyIncome: {dict_data['MonthlyIncome']}
    6) NumberOfOpen: {dict_data['NumberOfOpen']}
    7) NumberOfTimes90: {dict_data['NumberOfTimes90']}
    8) NumberRealEstate: {dict_data['NumberRealEstate']}
    9) NumberOfTime60: {dict_data['NumberOfTime60']}
    10) NumberOfDependents: {dict_data['NumberOfDependents']}
    """
    )

    # evaluate and return prediction (text)
    button_ok = st.button("Predict")
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f"## {output[0]}")
        st.success("Success!")


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Predict")
    if button_ok:
        # заглушка так как не выводим все предсказания
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_["predict"] = output.json()["prediction"]
        st.write(data_.head())
