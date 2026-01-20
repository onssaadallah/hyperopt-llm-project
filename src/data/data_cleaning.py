import pandas as pd
import numpy as np 
from pathlib import Path
class CleanData :
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()


    def clean(self, df=None):
        #remove column Co2  because it has a lot of missing values
        df = df if df is not None else self.df
        for col in df.columns:
            df = df[df[col] != -9999.0]
            df = df[df[col] != -9999.99]
        df = df[df['wv (m/s)'] >= 0]
        return df


    def interpolate(self, df=None, freq='10T'):
        df = df if df is not None else self.df
        df.index.name = "datetime"
        df = df[[c for c in df.columns if c != "CO2 (ppm)"]]
        df_resampled = df.groupby(pd.Grouper(freq=freq)).mean()
        missing_before = df_resampled.isna().any(axis=1).sum()
        print(f"==> {missing_before} rows contain missing values and will be interpolated <==")
        df_filled = df_resampled.interpolate().round(2)
        df_filled = df_filled.fillna(method='bfill').fillna(method='ffill')
        return df_filled
    

    
    def feature_selection(self, name: str, df=None):
        # If df not provided, use the class DataFrame
        df = df if df is not None else self.df
        # Select relevant features
        usecols = ['T (degC)', 'rh (%)', 'p (mbar)', 'wv (m/s)']
        weather = df[usecols]
        # Save the clean data
        base_dir = Path("C:/Users/user.IBRAHIM-IK-SZHE/Meta_LLM_HPO/hyperopt-llm-project/data/clean_data")
        save_path = base_dir / f"{name}.csv"
        weather.to_csv(save_path)
        return weather