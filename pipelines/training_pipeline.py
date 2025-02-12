from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig


@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    config = ModelNameConfig(model_name="LinearRegression")
    x_train,x_test,y_train,y_test =clean_df(df)
    model = train_model(x_train,x_test,y_train,y_test,config)
    r2_score,rmse = evaluate_model(model,x_test,y_test)
