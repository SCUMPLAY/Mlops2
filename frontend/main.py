"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os

import yaml
import streamlit as st
from src.data.get_data import load_data, get_dataset
from src.plotting.charts import barplot_group, kdeplotting, scatter_plot, bar_plot
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input, evaluate_from_file

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        "https://techcrunch.com/wp-content/uploads/2015/11/shutterstock_302406614.png",
        width=600,
    )

    st.markdown("# Описание проекта")
    st.title("MLOps project: Credit scoring challenge to identify who will experience financial hardship in the next two years 🌐🏦 ")
    st.write(
        """
        Алгоритмы кредитного скоринга, которые предполагают вероятность дефолта, — это метод,
        который банки используют для определения того, следует ли предоставлять кредит. Этот конкурс требует, 
        чтобы участники улучшили состояние дел в кредитном скоринге,предсказывая вероятность того, что кто-то испытает финансовые затруднения в следующие два года.
        Целью этого конкурса является создание модели, которую заемщики смогут использовать для принятия наилучших финансовых решений.
"""
    )

    # name of the columns
    st.markdown(
        """
        ### Описание полей 
            - Id -- Индексная колонка, представляющая уникальный идентификатор каждой записи.
            - SeriousDlqin2yrs -- Бинарный признак, указывающий, имел ли заемщик серьезную просрочку платежей в течение двух лет.    Значение 1 означает наличие просрочки, а значение 0 - отсутствие просрочки.
            - Revolving -- Доля использования необеспеченных кредитных линий. Это отношение суммы задолженности по кредитным картам к лимитам по кредитным картам.
            - age -- Возраст заемщика.
            - NumberOfTime30 -- Количество раз, когда заемщик имел просрочку платежей на 30-59 дней, но не хуже.
            - DebtRatio -- Соотношение долга к доходу заемщика.
            - MonthlyIncome -- Ежемесячный доход заемщика.
            - NumberOfOpen  -- Общее количество открытых кредитных линий и кредитов.
            - NumberOfTimes90 -- Количество раз, когда заемщик имел просрочку платежей на 90+ дней.
            - NumberRealEstate -- Количество ипотечных кредитов и кредитных линий на недвижимость.
            - NumberOfTime60 -- Количество раз, когда заемщик имел просрочку платежей на 60-89 дней, но не хуже.
            - NumberOfDependents -- Количество иждивенцев заемщика (супруг, дети и т.д.).
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load and write dataset
    data = get_dataset(dataset_path=config["preprocessing"]["train_path"])
    st.write(data.head())

    # plotting with checkbox
    age_seriousdlqin2yrs = st.sidebar.checkbox("Возраст-К выдачи кредита")
    revolving_seriousdlqin2yrs = st.sidebar.checkbox("Необеспеченные кредитные линии-К выдачи кредита")
    numberofdependents_seriousdlqin2yrs = st.sidebar.checkbox("Количество иждивенцов -К выдачи кредита")
    numberrealestatel_seriousdlqin2yrs = st.sidebar.checkbox("Кредитные линии на недвижимости ")

    if age_seriousdlqin2yrs:
        st.write("**Гипотеза №1** -Возраст хорших заемщиков больше, по сравнению с плохими (распределния возраста в зависимости от флага дефолта смещено в большую строну при Targe =0),возратные более платежеспособные")
        st.pyplot(
            kdeplotting(
                data=data,
                data_x="age",
                hue="SeriousDlqin2yrs",
                title="Возраст-К выдачи кредита",
            )

        )
        st.write("- Вывод по данной гипотезы, тем страше люди тем боллее они платежоспособны. ")
    if revolving_seriousdlqin2yrs:
        st.write("**Гипотеза №2** -Высокая доля использования необеспеченных кредитных линий (RevolvingUtilizationOfUnsecuredLines) "
                 "связана с большей вероятностью серьезной просрочки платежей (SeriousDlqin2yrs). Пользователи, у которых сумма задолженности "
                 "по кредитным картам близка или превышает предоставленные кредитные лимиты, могут испытывать трудности с возвратом долга.")
        st.pyplot(
            scatter_plot(
                X=data["Revolving"],
                y=data["SeriousDlqin2yrs"],
                x_label="Revolving",
                y_label="SeriousDlqin2yrs",
                title="Необеспеченные кредитные линии-К выдачи кредита",
            )
        )
        st.write("- Визуальный анализ показывает, что существует некоторая связь между долей "
                 "использования необеспеченных кредитных линий и серьезной просрочкой платежей. Заемщики с более высокой долей использования необеспеченных кредитных линий (близкой к 1) имеют большую вероятность столкнуться с серьезной просрочкой платежей. С другой стороны, "
                 "заемщики с более низкой долей использования необеспеченных кредитных линий (близкой к 0) имеют меньшую вероятность серьезной просрочки платежей.")
    if numberofdependents_seriousdlqin2yrs:
        st.write("**Гипотеза №3** Большее количество иждивенцев (NumberOfDependents) может быть связано с более низкой вероятностью серьезной просрочки платежей. "
                 "Заемщики с семейными обязательствами могут быть более ответственными и стараться выполнять свои финансовые обязательства.")
        st.pyplot(
            barplot_group(
                data=data,
                col_main="NumberOfDependents",
                col_group="SeriousDlqin2yrs",
                title="Количество иждивенцов -К выдачи кредита",
            )
        )
        st.write("- Гипотеза потвердилась , люди у которых есть семейные обязательства болле отвествные и страяются выполнять свои обязательства.")
    if numberrealestatel_seriousdlqin2yrs:
        st.write("**Гипотеза №4** Наличие ипотечных кредитов и кредитных линий на недвижимость (NumberRealEstateLoansOrLines) может указывать на более надежных заемщиков с меньшей вероятностью серьезной просрочки платежей. "
                 "Имущество может служить в качестве обеспечения и повышать ответственность заемщика. Визалзуруй данную гипотезу")
        st.pyplot(
            bar_plot(
                data_grouped=data.groupby('NumberRealEstate')['SeriousDlqin2yrs'].mean(),
                x_label="Количество кредитов на недвижимость или линий",
                y_label="Вероятность серьезного правонарушения",
                title="Вероятность серьезной просроченной задолженности по количеству кредитов на недвижимость или линий",
            )
        )
        st.write("Диаграмма показывает, что заемщики с наличием ипотечных кредитов и кредитных линий на недвижимость (NumberRealEstateLoansOrLines) имеют более низкую вероятность серьезной просрочки платежей. Заемщики с большим количеством ипотечных кредитов и кредитных линий на недвижимость обычно являются более надежными и ответственными плательщиками. "
                 "Возможно, наличие недвижимости в качестве обеспечения повышает их финансовую ответственность и способность выполнять финансовые обязательства своевременно.")


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model LightGBM")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["preprocessing"]["unique_values_path"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "xlsx"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory data analysis": exploratory,
        "Training model": training,
        "Prediction": prediction,
        "Prediction from file": prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
