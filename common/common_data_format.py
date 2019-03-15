import pandas as pd

class TimeSeriesData:
    """Base class for all time series data.
    """
    def __init__(
        self,
        time_col_name, target_col_name, 
        id_col_names, static_fea_names, 
        dynamic_fea_names, frequency, 
        time_format, description=None,
        df=None, data_prep_function=None,
        data_prep_function_args=None
        ):
        """Initialize a time series data object.

        Args:
            time_col_name (str): name of the timestamp column
            target_col_name (str): name of the target column that need to be forecasted
            id_col_names (list): names of the columns for identifying a unique time series
            static_fea_names (list): names of the feature columns that do not change over time
            dynamic_fea_names (list): names of the feature columns that change over time
            frequency (str): frequency of the timestamps represented by the time series offset
                             aliases used in Pandas (e.g. "W" for weekly frequency)
                             See https://pandas.pydata.org/pandas-docs/stable/user_guide 
                             /timeseries.html#timeseries-offset-aliases for details
            time_format (str): format of the timestamps (e.g., "%d.%m.%Y %H:%M:%S")
            description (str): description of the data (e.g., "training set", "testing set")
            df (Pandas DataFrame): input time series dataframe
            data_prep_function (obj): Python function object that usually downloads and 
                                      prepares the time series dataset. The function should 
                                      return a Pandas DataFrame.
            data_prep_function_args (dict): a dictionary including the input argument values 
                                            of the data_pre_function.

            Note that neither static_fea_names or dynamic_fea_names should include the timestamp
            column and the target column. 
        """
        if df is None:
            try:
                df = self._get_ts_dataframe(data_prep_function, data_prep_function_args)
            except:
                raise ValueError("Invalid data preparation function or arguments of the function.")
        else:
            print(len(df))
            if len(df) == 0:
                raise ValueError("Input time series dataframe should not be empty.")
        self.df = df

        self._check_input_columns("timestamp", time_col_name)
        self._check_input_columns("target", target_col_name)
        self._check_input_columns("name_list", id_col_names)
        self._check_input_columns("name_list", static_fea_names)
        self._check_input_columns("name_list", dynamic_fea_names)
        self.time_col_name = time_col_name
        self.target_col_name = target_col_name
        self.id_col_names = id_col_names
        self.static_fea_names = static_fea_names
        self.dynamic_fea_names = dynamic_fea_names

        self._check_frequency(frequency)
        self._check_time_format(time_format)
        self.frequency = frequency
        self.time_format = time_format
        self.description = description

    def _get_ts_dataframe(self, data_prep_function, data_prep_function_args):
        """Get time series dataframe using the customized data preparation function.
        """
        return data_prep_function(**data_prep_function_args)

    def _check_input_columns(self, input_type, input_col_names):   
        """Check if input column/feature names are valid.
        """
        df_col_names = list(self.df)
        if input_type in ["timestamp", "target"]:
            assert isinstance(input_col_names, str)
            if input_col_names not in df_col_names:
                raise ValueError("Invalid {} column name. It cannot be found in the input dataframe.".format(input_type)) 
        else: 
            assert isinstance(input_col_names, list)
            for c in input_col_names:
                if c not in df_col_names:
                    raise ValueError(c + " is an invalid column name. It cannot be found in the input dataframe.")
     
    def _check_frequency(self, frequency):
        """Check if the data frequency is valid.
        """        
        try:
            pd.date_range(self.df[self.time_col_name][0], periods=3, freq=frequency)
        except:
            raise ValueError("Input data frequency is invalid. Please use the aliases in " +
                             "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases")

    def _check_time_format(self, time_format):
        """Check if the timestamp format is valid.
        """   
        try:
            pd.to_datetime(self.df[self.time_col_name], format=time_format)
        except:
            raise ValueError("Incorrect date format is specified.")
    
    def _auto_detect_frequency(self):
        """Automatically detect data frequency if it is not specified.
        """
        pass


    @property
    def config(self):
        return {"time_col_name": self.time_col_name,
                "target_col_name": self.target_col_name,
                "id_col_names": self.id_col_names,
                "static_fea_names": self.static_fea_names,
                "dynamic_fea_names": self.dynamic_fea_names,
                "frequency": self.frequency,
                "time_format": self.time_format,
                "description": self.description
                }

    @config.setter
    def config(self, new_config):
        for key, value in new_config.items():
            setattr(self, key, value)

    def data_quality_check(self):
        """Check data quality.
        """
        pass

    def split_data(self, mode):
        """Split the data for cross validation or other purposes.
        """
        pass

def dummy_data_prep_function(df, n):
    print(n)
    return df

if __name__ == "__main__":
    sales = {"timestamp": ["01/01/2001", "02/01/2001"], "sales": [1234, 2345],  
             "store": ["1001", "1002"], "brand": ["1", "2"], 
             "income": [53000, 65000], "price": [10, 12]}
    # Option 1
    df = pd.DataFrame(sales)
    # Option 2
    data_prep_function = dummy_data_prep_function
    data_prep_function_args = {"df": df, "n": 2019}

    time_col_name = "timestamp"
    target_col_name = "sales"
    id_col_names = ["store", "brand"]
    static_fea_names = id_col_names + ["income"]
    dynamic_fea_names = ["price"]
    frequency = "MS" #"monthly"
    time_format = "%m/%d/%Y"
    ts_data = TimeSeriesData(time_col_name, target_col_name, id_col_names, \
                             static_fea_names, dynamic_fea_names, frequency, time_format, df=None, \
                             data_prep_function=data_prep_function, data_prep_function_args=data_prep_function_args)
    print(ts_data.df)

    ts_data.time_col_name = "timestamp_new"
    print(ts_data.config)

    ts_data.config = {"time_col_name": "timestamp"}
    print(ts_data.config)