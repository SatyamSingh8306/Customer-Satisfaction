import logging
import pandas as pd
import numpy as np
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow 
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train:pd.DataFrame,
    x_test : pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config :ModelNameConfig,
    )->RegressorMixin:
    """
    train the model on the ingest data
    Args:
        x_train:Trainig Data
        x_test:Testing Data
        y_train: Training Data
        y_test:Testing Data
    """
    try:
        model = None
        if config.model_name=="LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(x_train,y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error("Error in training data {e}")
        raise e
