import pandas as pd

def specify_data_schema(
        df, time_col_name, target_col_name, 
        id_col_names, static_fea_names, 
        dynamic_fea_names, frequency, 
        time_format, description=None
        ):
        """Specify the schema of a time series dataset.

        Args:
            df (Pandas DataFrame): input time series dataframe
            time_col_name (str): name of the timestamp column
            target_col_name (str): name of the target column that need to be forecasted
            id_col_names (list): names of the columns for identifying a unique time series of
                                 the target variable
            static_fea_names (list): names of the feature columns that do not change over time
            dynamic_fea_names (list): names of the feature columns that can change over time
            frequency (str): frequency of the timestamps represented by the time series offset
                             aliases used in Pandas (e.g. "W" for weekly frequency). Please see
                             https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases 
                             for details.
            time_format (str): format of the timestamps (e.g., "%d.%m.%Y %H:%M:%S")
            description (str): description of the data (e.g., "training set", "testing set")

            Note that neither static_fea_names nor dynamic_fea_names should include the timestamp
            column and the target column. 

        Returns:
            df_config (dict): configuration of the time series data 
        """
        if len(df) == 0:
            raise ValueError("Input time series dataframe should not be empty.")

        df_col_names = list(df)  
        _check_col_names(df_col_names, time_col_name, "timestamp")
        _check_col_names(df_col_names, target_col_name, "target")
        _check_col_names(df_col_names, id_col_names, "name_list")
        _check_col_names(df_col_names, static_fea_names, "name_list")
        _check_col_names(df_col_names, dynamic_fea_names, "name_list")
        _check_frequency_format(df, time_col_name, frequency)
        _check_time_format(df, time_col_name, time_format)
        _check_static_fea(df, id_col_names, static_fea_names)

        # Configuration of the time series data
        df_config = {"time_col_name": time_col_name,
                     "target_col_name": target_col_name,
                     "id_col_names": id_col_names,
                     "static_fea_names": static_fea_names,
                     "dynamic_fea_names": dynamic_fea_names,
                     "frequency": frequency,
                     "time_format": time_format,
                     "description": description
                    }
        return df_config

def _check_col_names(df_col_names, input_col_names, input_type):   
    """Check if input column/feature names are valid.
    """
    if input_type in ["timestamp", "target"]:
        assert isinstance(input_col_names, str)
        if input_col_names not in df_col_names:
            raise ValueError("Invalid {} column name. It cannot be found in the input dataframe.".format(input_type)) 
    else: 
        assert isinstance(input_col_names, list)
        for c in input_col_names:
            if c not in df_col_names:
                raise ValueError(c + " is an invalid column name. It cannot be found in the input dataframe.")

def _check_frequency_format(df, time_col_name, frequency):
    """Check if the data frequency is valid.
    """        
    try:
        pd.date_range(df[time_col_name][0], periods=3, freq=frequency)
    except:
        raise ValueError("Input data frequency is invalid. Please use the aliases in " +
                         "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases")

def _check_time_format(df, time_col_name, time_format):
    """Check if the timestamp format is valid.
    """   
    try:
        pd.to_datetime(df[time_col_name], format=time_format)
    except:
        raise ValueError("Incorrect date format is specified.")

def _check_static_fea(df, id_col_names, static_fea_names):
    """Check if the input static features change over time and include id_col_names.
    """ 
    for fea in static_fea_names:
        if df.groupby(id_col_names)[fea].nunique().max() > 1:
            raise ValueError("Input feature column {} is supposed to be static but it is not.".format(fea))
    if not set(id_col_names) <= set(static_fea_names):
        raise ValueError("Static features do not include all the columns necessary for uniquely specifying each target time series.")
    
if __name__ == "__main__":
    sales = {"timestamp": ["01/01/2001", "02/01/2001", "02/01/2001"], "sales": [1234, 2345, 1324],  
            "store": ["1001", "1002", "1001"], "brand": ["1", "2", "1"], 
            "income": [53000, 65000, 53000], "price": [10, 12, 11]}
    df = pd.DataFrame(sales)
    time_col_name = "timestamp"
    target_col_name = "sales"
    id_col_names = ["store", "brand"]
    static_fea_names = id_col_names + ["income"]
    dynamic_fea_names = ["price"]
    frequency = "MS" #monthly start
    time_format = "%m/%d/%Y"
    df_config = specify_data_schema(df, time_col_name, target_col_name, id_col_names, \
                                    static_fea_names, dynamic_fea_names, frequency, time_format)
    print(df_config)