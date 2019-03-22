from abc import abstractmethod
import pandas as pd
import numpy as np

from functools import reduce

from base_ts_estimators import BaseTSFeaturizer
from common.utils import is_datetime_like, convert_to_tsdf


class BaseLagFeaturizer(BaseTSFeaturizer):
    """
    Base abstract  lag featurizer class for all lag featurizers.
    Child classes need to implement _lag_single_ts, which creates a data
    frame with lag features on a single time series.
    """
    @property
    def input_col_names(self):
        return self._input_col_names

    @input_col_names.setter
    def input_col_names(self, val):
        if isinstance(val, list):
            self._input_col_names = val
        else:
            self._input_col_names = [val]

    @property
    def max_test_timestamp(self):
        return self._max_test_timestamp

    @max_test_timestamp.setter
    def max_test_timestamp(self, val):
        if val is not None and not is_datetime_like(val):
            self._max_test_timestamp = \
                pd.to_datetime(val, format=self.time_format)
        else:
            self._max_test_timestamp = val

    @property
    def future_value_available(self):
        return self._future_value_available

    @future_value_available.setter
    def future_value_available(self, val):
        if not val and not self.max_test_timestamp:
            raise Exception('max_test_timestamp must be set when '
                            'future_value_available is False')
        self._future_value_available = val

    @property
    def training_df(self):
        return self._training_df

    @training_df.setter
    def training_df(self, val):
        self._training_df = val

    @abstractmethod
    def _lag_single_ts(self, input_df, forecast_creation_time):
        """
        Creates lag features on a data frame containing a single time series.
        Args:
        input_df(pandas.DataFrame): Input data frame with a time column
            and columns to create lag features on.
        forecast_creation_time(pandas.datetime): A timestamp
            specifying when the lag features are created.

        Returns:
            pandas.DataFrame: Data frame with time column and lag features.
        """
        pass

    def _create_lag_df(self, input_df, lags, frequency):
        """
        Creates a data frame with lags.
        Args:
            input_df(pandas.DataFrame): Input data frame with a time column
                and columns to create lag features on.
            lags(int or list of int): Lags to compute.
            frequency(str): Frequency of the shift operation.
                See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

        Returns:
            pandas.DataFrame: Data frame with time column and lag features.
        """
        df_list = [input_df]
        for lag in lags:
            df_shifted = input_df[self.input_col_names] \
                .shift(lag, freq=frequency)
            df_shifted.columns = [x + '_lag' + str(lag) for x in
                                  df_shifted.columns]
            df_list.append(df_shifted)
        lag_df = reduce(
            lambda left, right:
            pd.merge(left, right, how='left',
                     left_index=True, right_index=True),
            df_list)
        lag_df.drop(self.input_col_names, inplace=True, axis=1)
        return lag_df

    def fit(self, X, y=None):
        """
        To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this transformer
        """
        return self

    def transform(self, X):
        """
        Creates lag features on a subset of columns in X.

        Args:
             X(pandas.DataFrame): Input data frame to create lag features on.
        Returns:
            pandas.DataFrame: Output data frame with the lag features added
                to the input data frame.
        """
        self._check_config_cols_exist(X)

        col_names = [self.time_col_name] + self.input_col_names + \
                    self.ts_id_col_names
        merge_col_names = [self.time_col_name] + self.ts_id_col_names

        if self.training_df is not None:
            self._check_config_cols_exist(self.training_df)
            forecast_creation_time = self.training_df[self.time_col_name].max()
            X_tmp = pd.concat([self.training_df, X], sort=True)
            X_tmp = X_tmp[col_names].copy()
        else:
            if not self.future_value_available:
                # Compute an imaginary forecast creation time for the training
                # data based on the maximum timestamp to forecast on
                max_train_timestamp = X[self.time_col_name].max()
                train_test_timestamp_diff = \
                    self.max_test_timestamp - max_train_timestamp
                forecast_creation_time = \
                    max_train_timestamp - train_test_timestamp_diff
            else:
                forecast_creation_time = X[self.time_col_name].max()
            X_tmp = X[col_names].copy()

        if self.ts_id_col_names is None:
            X_lag_tmp = self._lag_single_ts(X_tmp, forecast_creation_time)
        else:
            X_lag_tmp = X_tmp.groupby(self.ts_id_col_names).\
                apply(lambda g: self._lag_single_ts(g, forecast_creation_time))

            X_lag_tmp.reset_index(inplace=True)

        if self.training_df is not None:
            X_lag_tmp = X_lag_tmp.loc[X_lag_tmp[self.time_col_name] >
                                      forecast_creation_time].copy()
        X = pd.merge(X, X_lag_tmp, on=merge_col_names)

        if X.shape[0] == 0:
            raise Exception('The featurizer output is empty. Set the '
                            'training_df property of the featurizer to '
                            'None if transforming training data.')
        return X


class BasePeriodicLagFeaturizer(BaseLagFeaturizer):
    """
    Base abstract class for periodical lag features.
    Periodical lag features are features that from the same time point of
    previous periods, e.g. same week of year of previous years. Multiple
    lags are aggregated to compute a single lag feature. For example,
    when using 5 years of history data, the 5 values from the same week of
    each of the 5 years are aggregate.
    Child classes need to implement _calculate_lags.
    """
    @abstractmethod
    def _calculate_lags(self):
        """
        Computes and returns lags, as integer or list of integers, needed by a
        particular featurizer.
        """
        pass

    def _lag_single_ts(self, input_df, forecast_creation_time):
        """
        Creates lag features on a data frame containing a single time series.

        Args:
        input_df(pandas.DataFrame): Input data frame with a time column
            and columns to create lag features on.
        forecast_creation_time(pandas.datetime): A timestamp
            specifying when the lag features are created.

        Returns:
            pandas.DataFrame: Data frame with time column and lag features.
        """
        input_df = convert_to_tsdf(input_df,
                                   time_col_name=self.time_col_name,
                                   time_format=self.time_format,
                                   frequency=self.frequency)

        output_df = pd.DataFrame({
            self.time_col_name: input_df.index.get_level_values(0)})
        min_time_stamp = output_df[self.time_col_name].min()
        max_time_stamp = output_df[self.time_col_name].max()

        if not self.future_value_available:
            input_df.loc[input_df.index.get_level_values(0) >
                         forecast_creation_time, self.input_col_names] = np.nan

        lag_all = self._calculate_lags()

        lag_all = [lag for lag in lag_all
                   if (max_time_stamp -
                       pd.to_timedelta(lag, self._lag_frequency)) >=
                   min_time_stamp]

        lag_df = self._create_lag_df(input_df, lags=lag_all,
                                     frequency=self._lag_frequency)
        for col in self.input_col_names:
            lag_cols = [c for c in lag_df.columns if c.startwith(col)]
            output_col_name = col + '_' + self.output_col_suffix

            output_df[output_col_name] = lag_df[lag_cols].apply(
                self.agg_func, axis=1, **self.agg_args)

        return output_df


class LagFeaturizer(BaseLagFeaturizer):
    """
    Computes regular lag features.
    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        input_col_names(str or list of str): Names of the columns to create the
            lag feature on.
        lags(int or list of int): Lags to compute. When
            future_value_available is False, lags must be positive.
        future_value_available(bool): Whether future values of the input
            columns are available at the forecast creation time. Default
            value is False.
        max_test_timestamp(pandas.datetime): Maximum timestamp of the testing
            data to generate forecasting on. This value is needed to prevent
            creating lag features on the training data that are not available
            for the testing data. For example, the features and models are
            created on week 7 to forecast week 8 to week 10. It would not make
            sense to create a lag feature using data from week 8 and
            week 9, because they are not available at the forecast creation
            time. Thus, it does not make sense to create a lag
            feature using data from week 5 and week 6 for week 7.
        training_df(pandas.DataFrame): Training data needed to compute lag
            features on testing data.
            Note: this property must be None when transforming the
            training data, and training_df can only be passed after
            transforming the training data.  It's not recommended to save a
            pipeline with training_df not set to None, because it results in a
            large pipeline object, especially when you have multiple
            featurizers requiring the training data at scoring time.
            To set this value on in a pipeline, use the following code
            pipeline.set_params('featurizer_step_name__train_df) = None
            pipeline.set_params('featurizer_step_name__train_df) = train_df
            featurizer_step_name is the name of the featurizer step when
            creating the pipeline.
    """
    def __init__(self, df_config, input_col_names, lags,
                 future_value_available=False, max_test_timestamp=None,
                 training_df=None):
        super(LagFeaturizer, self).__init__(df_config)

        self.input_col_names = input_col_names
        self.lags = lags
        self.max_test_timestamp = max_test_timestamp
        self.future_value_available = future_value_available

        if not self.future_value_available:
            for lag in lags:
                if lag < 0:
                    raise Exception('lag can not be negative when '
                                    'future_value_available is False')

        self.training_df = training_df

    def _lag_single_ts(self, input_df, forecast_creation_time):
        """
        Creates lag features on a data frame containing a single time series.
        Args:
        input_df(pandas.DataFrame): Input data frame with a time column
            and columns to create lag features on.
        forecast_creation_time(pandas.datetime): A timestamp
            specifying when the lag features are created.

        Returns:
            pandas.DataFrame: Data frame with time column and lag features.
        """
        input_df = convert_to_tsdf(input_df,
                                   time_col_name=self.time_col_name,
                                   time_format=self.time_format,
                                   frequency=self.frequency
                                   )

        if not self.future_value_available:
            input_df.loc[input_df.index.get_level_values(0) >
                         forecast_creation_time, self.input_col_names] = np.nan

        lag_df = self._create_lag_df(input_df, lags=self.lags,
                                     frequency=self.frequency)
        lag_df.drop(self.ts_id_col_names, inplace=True, axis=1)

        return lag_df


class SameWeekOfYearLagFeaturizer(BasePeriodicLagFeaturizer):
    """
    Creates a lag feature based on data from the same week of previous years.

    This feature is useful for data with daily, weekly, and yearly
    seasonalities and has hourly, daily, or weekly frequency. It is computed
    by aggregating values of and around the same week of previous years.
    For hourly data, value of each week is taken from the same hour of day and
    the same day of week.
    For daily data, value of each week is taken from the same day of week.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        input_col_names(str or list of str): Names of the columns to create the
            lag feature on.
        training_df(pd.DataFrame): Training data needed to compute lag
            features on testing data.
            Note: this property must be None when transforming the
            training data, and training_df can only be passed after
            transforming the training data.  It's not recommended to save a
            pipeline with training_df not set to None, because it results in a
            large pipeline object, especially when you have multiple
            featurizers requiring the training data at scoring time.
            To set this value on in a pipeline, use the following code
            pipeline.set_params('featurizer_step_name__train_df) = None
            pipeline.set_params('featurizer_step_name__train_df) = train_df
            featurizer_step_name is the name of the featurizer step when
            creating the pipeline.
        n_years(int): Number of previous years data to use. Default value is 3.
        week_window(int): Number of weeks before and after the same week of
            year to use, which should help reduce noise in the data. Default
            value is 1.
        agg_func(function or str): Aggregation function to apply on multiple
            previous values. Default value is 'mean'. Other commonly used
            values are 'std', 'quantile'.
        agg_args(dict): Additional arguments passed to the aggregation function.
        future_value_available(bool): Whether future values of the input
            columns are available at the forecast creation time. Default
            value is False.
        max_test_timestamp(pandas.datetime): Maximum timestamp of the testing
            data to generate forecasting on. This value is needed to prevent
            creating lag features on the training data that are not available
            for the testing data. For example, the features and models are
            created on week 7 to forecast week 8 to week 10. It would not make
            sense to create a lag feature using data from week 8 and
            week 9, because they are not available at the forecast creation
            time. Thus, it does not make sense to create a lag
            feature using data from week 5 and week 6 for week 7.
            When future_value_available is False, max_test_timestamp must be
            specified.
        output_col_suffix(str): Name suffix of the output lag feature column.
            Default value is 'same_woy_lag'.
    """

    def __init__(self, df_config, input_col_names, training_df=None, n_years=3,
                 week_window=1, agg_func='mean', agg_args={},
                 future_value_available=False, max_test_timestamp=None,
                 output_col_suffix='same_woy_lag'):
        super(SameWeekOfYearLagFeaturizer, self).__init__(df_config)

        self.input_col_names = input_col_names
        self.n_years = n_years
        self.week_window = week_window
        self.agg_func = agg_func
        self.agg_args = agg_args
        self.max_test_timestamp = max_test_timestamp
        self.future_value_available = future_value_available
        self.output_col_suffix = output_col_suffix

        self.training_df = training_df

        self._lag_frequency = 'W'

    def _calculate_lags(self):
        """Calculate of list of numbers of weeks to lag."""
        week_lag_base = 52
        week_lag_last_year = list(range(week_lag_base - self.week_window,
                                        week_lag_base + self.week_window + 1))
        week_lag_all = []
        for i in range(self.n_years):
            week_lag_all += [j + i * 52 for j in week_lag_last_year]

        return week_lag_all


class SameDayOfYearLagFeaturizer(BasePeriodicLagFeaturizer):

    """
    Creates a lag feature based on data from the same day of previous years.

    This feature is useful for data with daily and yearly seasonalities and
    has hourly or daily frequency. It is computed by aggregating values of and
    around the same day of year of previous years. For hourly data, value of
    each day is taken from the same hour of day.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        input_col_names(str or list of str): Names of the columns to create the
            lag feature on.
        training_df(pd.DataFrame): Training data needed to compute lag
            features on testing data.
            Note: this property must be None when transforming the
            training data, and training_df can only be passed after
            transforming the training data.  It's not recommended to save a
            pipeline with training_df not set to None, because it results in a
            large pipeline object, especially when you have multiple
            featurizers requiring the training data at scoring time.
            To set this value on in a pipeline, use the following code
            pipeline.set_params('featurizer_step_name__train_df) = None
            pipeline.set_params('featurizer_step_name__train_df) = train_df
            featurizer_step_name is the name of the featurizer step when
            creating the pipeline.
        n_years(int): Number of previous years data to use. Default value is 3.
        day_window(int): Number of days before and after the same day  of
            year to use, which should help reduce noise in the data. Default
            value is 1.
        agg_func(function or str): Aggregation function to apply on multiple
            previous values. Default value is 'mean'. Other commonly used
            values are 'std', 'quantile'.
        agg_args(dict): Additional arguments passed to the aggregation function.
        future_value_available(bool): Whether future values of the input
            columns are available at the forecast creation time. Default
            value is False.
        max_test_timestamp(pandas.datetime): Maximum timestamp of the testing
            data to generate forecasting on. This value is needed to prevent
            creating lag features on the training data that are not available
            for the testing data. For example, the features and models are
            created on week 7 to forecast week 8 to week 10. It would not make
            sense to create a lag feature using data from week 8 and
            week 9, because they are not available at the forecast creation
            time. Thus, it does not make sense to create a lag
            feature using data from week 5 and week 6 for week 7.
            When future_value_available is False, max_test_timestamp must be
            specified
        output_col_suffix(str): Name suffix of the output lag feature column.
            Default value is 'same_doy_lag'.

    """

    def __init__(self, df_config, input_col_names, training_df=None,
                 n_years=3, day_window=1, agg_func='mean', agg_args={},
                 future_value_available=False, max_test_timestamp=None,
                 output_col_suffix='same_doy_lag'):
        super(SameDayOfYearLagFeaturizer, self).__init__(df_config)
        self.input_col_names = input_col_names
        self.n_years = n_years
        self.day_window = day_window
        self.agg_func = agg_func
        self.agg_args = agg_args
        self.max_test_timestamp = max_test_timestamp
        self.future_value_available = future_value_available
        self.output_col_suffix = output_col_suffix

        self.training_df = training_df

        self._lag_frequency = 'D'

    def _calculate_lags(self):
        """Calculates a list of numbers of days to lag."""
        day_lag_base = 365
        day_lag_last_year = list(range(day_lag_base - self.day_window,
                                       day_lag_base + self.day_window + 1))
        day_lag_all = []
        for i in range(self.n_years):
            day_lag_all += [j + i * 365 for j in day_lag_last_year]

        return day_lag_all


## Temporary testing code
if __name__ == '__main__':
    tsdf = pd.DataFrame({'store': [1] * 10 + [2] * 10,
                         'date': list(pd.date_range('2011-01-01', '2011-01-10')) +
                                 list(pd.date_range('2011-01-01', '2011-01-10')),
                         'sales': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20]})

    df_config = {
        'time_col_name': 'date',
        'ts_id_col_names': 'store',
        'target_col_name': 'sales',
        'frequency': 'D',
        'time_format': '%Y-%m-%d'
    }

    lag_featurizer = LagFeaturizer(df_config, input_col_names='sales',
                                   lags=[1, 2, 3, 4],
                                   max_test_timestamp='2011-01-12')
    lag_featurizer.transform(tsdf)

