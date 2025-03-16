"""
Программа: Предобработка данных
Версия: 1.0
"""

import json
import warnings
import pandas as pd

warnings.filterwarnings("ignore")





def transform_types(data: pd.DataFrame, change_type_columns: dict) -> pd.DataFrame:
    """
    Преобразование признаков в заданный тип данных
    :param data: датасет
    :param change_type_columns: словарь с признаками и типами данных
    :return:
    """
    return data.astype(change_type_columns, errors="raise")


def get_bins(
    data: (int, float), first_val: (int, float), second_val: (int, float)
) -> str:
    """
    Генерация бинов для разных признаков
    :param data: датасет
    :param first_val: первый порог значения для разбиения на бины
    :param second_val: второй порог значения для разбиения на бины
    :return: датасет
    """
    assert isinstance(data, (int, float)), "Проблема с типом данных в признаке"
    result = (
        "small"
        if data <= first_val
        else "medium"
        if first_val < data <= second_val
        else "large"
    )
    return result


def check_columns_evaluate(data: pd.DataFrame, unique_values_path: str) -> pd.DataFrame:
    """
    Проверка на наличие признаков из train и упорядочивание признаков согласно train
    :param data: датасет test
    :param unique_values_path: путь до списока с признаками train для сравнения
    :return: датасет test
    """
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)

    column_sequence = unique_values.keys()

    assert set(column_sequence) == set(data.columns), "Разные признаки"
    return data[column_sequence]


def save_unique_train_data(
    data: pd.DataFrame, drop_columns: list, target_column: str, unique_values_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param drop_columns: список с признаками для удаления
    :param data: датасет
    :param target_column: целевая переменная
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    unique_df = data.drop(
        columns=drop_columns + [target_column], axis=1, errors="ignore"
    )
    # создаем словарь с уникальными значениями для вывода в UI
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


def pipeline_preprocess(data: pd.DataFrame, flg_evaluate: bool = True, **kwargs):
    """
    Пайплайн по предобработке данных
    :param data: датасет
    :param flg_evaluate: флаг для evaluate
    :return: датасет
    """
    data = data.drop(kwargs["drop_columns"], axis=1, errors="ignore")
    # проверка dataset на совпадение с признаками из train
    # либо сохранение уникальных данных с признаками из train
    if flg_evaluate:
        data = check_columns_evaluate(
            data=data, unique_values_path=kwargs["unique_values_path"]
        )
    else:
        save_unique_train_data(
            data=data,
            drop_columns=kwargs["drop_columns"],
            target_column=kwargs["target_column"],
            unique_values_path=kwargs["unique_values_path"],
        )



    assert isinstance(
        kwargs["map_bins_columns"], dict
    ), "Подайте тип данных для бинаризации в формате dict"
    # bins
    for key in kwargs["map_bins_columns"].keys():
        data[f"{key}_bins"] = data[key].apply(
            lambda x: get_bins(
                x,
                first_val=kwargs["map_bins_columns"][key][0],
                second_val=kwargs["map_bins_columns"][key][1],
            )
        )

   # transform types
    dict_category = {key: "category" for key in data.select_dtypes(["object"]).columns}
    data = transform_types(data=data, change_type_columns=dict_category)
    return data
