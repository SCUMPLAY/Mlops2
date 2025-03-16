"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def barplot_group(
    data: pd.DataFrame, col_main: str, col_group: str, title: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика boxplot
    :param data: датасет
    :param col_main: признак для анализа по col_group
    :param col_group: признак для нормализации/группировки
    :param title: название графика
    :return: поле рисунка
    """
    data_group = (
        data.groupby([col_group])[col_main]
        .value_counts(normalize=True)
        .rename("percentage")
        .mul(100)
        .reset_index()
        .sort_values(col_group)
    )

    data_group.columns = [col_group, col_main, "percentage"]

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 7))

    ax = sns.barplot(
        x=col_main, y="percentage", hue=col_group, data=data_group, palette="rocket"
    )
    for patch in ax.patches:
        percentage = "{:.1f}%".format(patch.get_height())
        ax.annotate(
            percentage,  # текст
            (
                patch.get_x() + patch.get_width() / 2.0,
                patch.get_height(),
            ),  # координата xy
            ha="center",  # центрирование
            va="center",
            xytext=(0, 10),
            textcoords="offset points",  # точка смещения относительно координаты
            fontsize=14,
        )
    plt.title(title, fontsize=20)
    plt.ylabel("Percentage", fontsize=14)
    plt.xlabel(col_main, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig


def kdeplotting(data: pd.DataFrame, data_x: str, hue: str, title: str) -> plt.Figure:
    """
    Отрисовка графика kdeplot
    :param data: датасет
    :param data_x: ось OX
    :param hue: группировка по признаку
    :param title: название графика
    :return: поле рисунка
    """
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

    sns.kdeplot(
        data=data[data[hue] == 0][data_x],
        common_norm=False,
        palette='rocket',
        ax=axes[0]
    )

    sns.kdeplot(
        data=data[data[hue] == 1][data_x],
        common_norm=False,
        palette='rocket',
        ax=axes[0]
    )

    axes[0].set_title(title, fontsize=16)
    axes[0].set_xlabel(data_x, fontsize=14)
    axes[0].set_ylabel("Target", fontsize=14)

    sns.boxplot(x=hue, y=data_x, data=data, palette='rocket', ax=axes[1])

    axes[1].set_title(f"Boxplot {data_x}", fontsize=16)
    axes[1].set_ylabel(data_x, fontsize=14)
    axes[1].set_xlabel("Target", fontsize=14)

    plt.show()
    return fig


def scatter_plot(X, y, x_label, y_label, title) -> matplotlib.figure.Figure:
    """
    Создание диаграммы разброса
    :param X: признак
    :param y: целевая переменная
    :param x_label: название оси X
    :param y_label: название оси Y
    :param title: заголовок графика
    """
    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return fig


def bar_plot(data_grouped, x_label, y_label, title) -> matplotlib.figure.Figure:
    """
    Создание столбчатой диаграммы для группированных данных
    :param data_grouped: группированные данные
    :param x_label: название оси X
    :param y_label: название оси Y
    :param title: заголовок графика
    :return: объект графика
    """
    fig, ax = plt.subplots()
    data_grouped.plot(kind='bar', ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()
    return fig
