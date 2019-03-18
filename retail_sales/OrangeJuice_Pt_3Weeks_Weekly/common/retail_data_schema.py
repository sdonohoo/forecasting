import os
import math
import datetime
import itertools
import pandas as pd
from benchmark_paths import *
from benchmark_settings import *
from common.data_schema import specify_data_schema

DEFAULT_TARGET_COL = "move"
DEFAULT_STATIC_FEA = ["store", "brand"]
DEFAULT_DYNAMIC_FEA = ["deal", "feat"]

def specify_retail_data_schema(
    sales = None,
    target_col_name = DEFAULT_TARGET_COL,
    static_fea_names = DEFAULT_STATIC_FEA,
    dynamic_fea_names = DEFAULT_DYNAMIC_FEA,
    description = None
    ):
    """Specify data schema of the retail dataset.

    Args:
        sales (Pandas DataFrame): sales data in the current forecast round
        target_col_name (str): name of the target column that need to be forecasted
        static_fea_names (list): names of the feature columns that do not change over time
        dynamic_fea_names (list): names of the feature columns that can change over time
        description (str): description of the data (e.g., "training set", "testing set")

    Returns:
        df_config (dict): configuration of the time series data 
        df (Pandas DataFrame): sales data combined with store demographic features
    """
    # Read all the sales data if it is not specified
    if sales is None:
        print("Sales dataframe is not given! All the sales data in OrangeJuice dataset will be used.")
        sales = pd.read_csv(os.path.join(DATA_DIR, "yx.csv"), index_col=False)

    # Read store demographic data
    storedemo = pd.read_csv(os.path.join(DATA_DIR, "storedemo.csv"), index_col=False)

    # Compute unit sales
    sales["move"] = sales["logmove"].apply(lambda x: round(math.exp(x)) if x > 0 else 0)

    # Make sure each time series has the same time span 
    store_list = sales["store"].unique()
    brand_list = sales["brand"].unique()
    week_list = range(sales["week"].min(), sales["week"].max() + 1)
    item_list = list(itertools.product(store_list, brand_list, week_list))
    item_df = pd.DataFrame.from_records(item_list, columns=["store", "brand", "week"])
    sales = item_df.merge(sales, how="left", on=["store", "brand", "week"])

    sales = item_df.merge(sales, how="left", on=["store", "brand", "week"])

    # Merge with storedemo
    df = sales.merge(storedemo, how="left", left_on="store", right_on="STORE")
    df.drop("STORE", axis=1, inplace=True)

    # Create timestamp
    df["timestamp"] = df["week"].apply(lambda x: FIRST_WEEK_START + datetime.timedelta(days=(x-1)*7))

    #print(sales.head())    
    df_config = specify_data_schema(df, time_col_name="timestamp", target_col_name=target_col_name, \
                                    id_col_names=["store", "brand"], static_fea_names=static_fea_names, \
                                    dynamic_fea_names=dynamic_fea_names, frequency="W", \
                                    time_format="%Y-%m-%d", description=description)
    return df_config, df

if __name__ == "__main__":
    df_config, sales = specify_retail_data_schema()
    print(df_config)