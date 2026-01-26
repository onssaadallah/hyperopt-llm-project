
import time 
import pandas as pd 
import numpy as np
from pathlib  import Path
import sys
sys.path.insert(0,r"C:Users/user.IBRAHIM-IK-SZHE/Meta_LLM_HPO/hyperopt-llm-project/src")
from data.data_cleaning  import CleanData
from data.data_preprocessing import DataPreprocessing
from models.model import BiLSTMForecast
from models.training import Trainer 
from pipelines.forecasting_pipeline import ForecastingPipeline



#--------------------------------------
 #Define Random Params 
#--------------------------------------
PARAMS = {
    "hidden_size": 5,
    "num_layers": 6,
    "dropout": 0.1,
    "batch_size": 128,
    "epochs": 10,
    "lag": 3,
    "kernel_size": 5,
    "optimizer": "adam",
    "lr": 1e-3,
}
#-------------------------------------
 #1.Clean Data
#-------------------------------------
RAW_TRAIN_PATH= "C:/Users/user.IBRAHIM-IK-SZHE/Meta_LLM_HPO\hyperopt-llm-project/data/raw_data/train.csv"
RAW_TEST_PATH = "C:/Users/user.IBRAHIM-IK-SZHE/Meta_LLM_HPO\hyperopt-llm-project/data/raw_data/eval.csv"
train_df = pd.read_csv(RAW_TRAIN_PATH,parse_dates=True, index_col="Date Time")
test_df = pd.read_csv(RAW_TEST_PATH,parse_dates=True,index_col ="Date Time")
#------Clean Raw Train Data-------------------
cl_train  = CleanData(train_df)
clean_train = cl_train.clean(train_df)
clean_train = cl_train.interpolate(clean_train)
clean_train = cl_train.feature_selection("clean_train",clean_train)
#------Clean Raw Test Data-------------------
cl_test  = CleanData(test_df)
clean_test = cl_test.clean(test_df)
clean_test  = cl_test.interpolate(clean_test)
clean_test= cl_test.feature_selection("clean_test",clean_test)
#--------------------------------------------------
 # 2. Load Clean Train/Test Data
#--------------------------------------------------

TARGET_COLS =['T (degC)', 'rh (%)', 'p (mbar)', 'wv (m/s)']


clean_train = pd.read_csv('C:/Users/user.IBRAHIM-IK-SZHE/Meta_LLM_HPO/hyperopt-llm-project/data/clean_data/clean_train.csv',
    parse_dates=["datetime"],
    index_col="datetime"
    )
clean_test = pd.read_csv('C:/Users/user.IBRAHIM-IK-SZHE/Meta_LLM_HPO/hyperopt-llm-project/data/clean_data/clean_test.csv', 
    parse_dates=["datetime"],
    index_col="datetime"
    )

#--------------------------
 #Define Forecasting Pipeline
#--------------------------
name_model = "Bi-LSTM"
experiment_name ="base_forecasting_Model"

# define ForecastingPipeline
# Create Base Forecasting Model 
#--------------------------
 #Define Forecasting Pipeline
#--------------------------
name_model = "Bi-LSTM"
experiment_name ="base_forecasting_Model"

# define ForecastingPipeline
pipeline = ForecastingPipeline(
    clean_train=clean_train,
    clean_test=clean_test,
    target_cols=TARGET_COLS,
    experiment_name=experiment_name
)
results = pipeline.run(name_model,PARAMS)