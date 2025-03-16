"""
Программа: Модель для прогнозирования того, кто столкнется с финансовыми трудностями в ближашие два года, и не сможет вернуть кредит.
Версия: 1.0
"""

import warnings
import optuna
import pandas as pd

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/params.yml"


class InsuranceCustomer(BaseModel):
    """
    Признаки для получения результатов модели
    """
    Revolving: float
    age: int
    MonthlyIncome: float
    NumberOfTime30: int
    DebtRatio: float
    NumberOfOpen: int
    NumberOfTimes90: int
    NumberRealEstate: int
    NumberOfTime60: int
    NumberOfDependents: float





@app.get("/hello")
def welcome():
    """
    Hello
    :return: None
    """
    return {'message': 'Hello Data Scientist!'}


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    # заглушка так как не выводим все предсказания, иначе зависнет
    return {"prediction": result[:5]}


@app.post("/predict_input")
def prediction_input(customer: InsuranceCustomer):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            customer.Revolving,
            customer.age,
            customer.MonthlyIncome,
            customer.NumberOfTime30,
            customer.DebtRatio,
            customer.NumberOfOpen,
            customer.NumberOfTimes90,
            customer.NumberRealEstate,
            customer.NumberOfTime60,
            customer.NumberOfDependents,
        ]
    ]

    cols = [
        "Revolving",
        "age",
        "MonthlyIncome",
        "NumberOfTime30",
        "DebtRatio",
        "NumberOfOpen",
        "NumberOfTimes90",
        "NumberRealEstate",
        "NumberOfTime60",
        "NumberOfDependents",
    ]

    data = pd.DataFrame(features, columns=cols)
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=data)[0]
    result = (
        {"The client is in arrears"}
        if predictions == 1
        else {"The client has no delay"}
        if predictions == 0
        else "Error result"
    )
    return result


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)
