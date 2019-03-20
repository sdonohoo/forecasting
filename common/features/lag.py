# from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from functools import reduce
from datetime import timedelta

from tsfeaturizer import BaseTSFeaturizer
from common.utils import is_datetime_like, convert_to_tsdf


class LagFeaturizer(BaseTSFeaturizer):
    def __init__(self, df_config, input_col_names, lags,
                 future_value_available=False, max_test_timestamp=None,
                 training_df=None):
        self.parse_tsdf_config(df_config)

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

    def _create_lag_df(self, input_df, lags, frequency):
        df_list = [input_df]
        for lag in lags:
            df_shifted = input_df[self.input_col_names]\
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

    def _lag_single_ts(self, input_df, forecast_creation_time):
        input_df = convert_to_tsdf(input_df,
                                   time_col_name=self.time_col_name,
                                   time_format=self.format,
                                   frequency=self.frequency
                                   )

        if not self.future_value_available:
            input_df.loc[input_df.index.get_level_values(0) >
                         forecast_creation_time, self.input_col_names] = np.nan

        lag_df = self._create_lag_df(input_df, self.lags, self.frequency)
        lag_df.drop(self.ts_id_col_names, inplace=True, axis=1)

        return lag_df

    def transform(self, X):
        col_names = [self.time_col_name] + self.input_col_names + \
                    self.ts_id_col_names
        merge_col_names = [self.time_col_name] + self.ts_id_col_names

        if self.training_df is not None:
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


class SameWeekDayHourLagFeaturizer(LagFeaturizer):
    """
    Creates a lag feature based on data from the same week of previous years.

    This feature is useful for data with daily, weekly, and yearly
    seasonalities and has hourly frequency. It is computed by calculating
    quantiles, mean, or std of values of and around the same week, same day
    of week, and same hour of day, of previous years.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        input_col_name(str): Name of the column to create the lag feature on.
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
        agg_func(str): Aggregation function to apply on multiple previous
            values, accepted values are 'mean', 'quantile', 'std'. Default
            value is 'mean'.
        q(float): Quantile to compute from previous values, if agg_func is
            'quantile', taking value between 0 and 1.
        output_col_name(str): Name of the output lag feature column.
            Default value is 'SameWeekDayHourLag'.
    """

    def __init__(self, df_config, input_col_names, training_df=None, n_years=3,
                 week_window=1, agg_func='mean', q=None,
                 future_value_available=False, max_test_timestamp=None,
                 output_col_suffix='same_woy_lag'):
        self.parse_tsdf_config(df_config)

        self.input_col_names = input_col_names
        self.max_test_timestamp = max_test_timestamp
        self.future_value_available = future_value_available
        self.training_df = training_df

        self.input_col_names = input_col_names
        self.n_years = n_years
        self.week_window = week_window
        self.agg_func = agg_func
        self.q = q
        self.output_col_suffix = output_col_suffix

    @property
    def training_df(self):
        return self._training_df

    @training_df.setter
    def training_df(self, val):
        self._training_df = val

    def _lag_single_ts(self, input_df, forecast_creation_time):
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

        week_lag_base = 52
        week_lag_last_year = list(range(week_lag_base - self.week_window,
                                        week_lag_base + self.week_window + 1))
        week_lag_all = []
        for i in range(self.n_years):
            week_lag_all += [j + i * 52 for j in week_lag_last_year]

        week_lag_all = [w for w in week_lag_all if
                        (max_time_stamp - timedelta(weeks=w)) >= min_time_stamp]

        lag_df = self._create_lag_df(input_df, week_lag_all, frequency='W')
        for col in self.input_col_names:
            week_lag_cols = [c for c in lag_df.columns if c.startwith(col)]
            output_col_name = col + '_' + self.output_col_suffix

            if self.agg_func == 'mean':
                output_df[output_col_name] = round(lag_df[
                    week_lag_cols].mean(
                    axis=1))
            elif self.agg_func == 'quantile' and self.q is not None:
                output_df[output_col_name] = \
                    round(lag_df[week_lag_cols].quantile(self.q, axis=1))
            elif self.agg_func == 'std':
                output_df[output_col_name] = round(lag_df[
                    week_lag_cols].std(
                    axis=1))

        return output_df


class SameDayHourLagFeaturizer(LagFeaturizer):

    """
    Creates a lag feature based on data from the same day of previous years.

    This feature is useful for data with daily and yearly seasonalities and
    has hourly frequency. It is computed by calculating quantiles, mean,
    or std of values of and around the same day of year and same hour of
    day of previous years.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        input_col_name(str): Name of the column to create the lag feature on.
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
        agg_func(int): Aggregation function to apply on multiple previous
            values, accepted values are 'mean', 'quantile', 'std'. Default
            value 'mean'.
        q(float): Quantile to compute from previous values, if agg_func is
            'quantile', taking value between 0 and 1.
        output_col_name(str): Name of the output lag feature column.
            Default value is 'SameDayHourLag'.

    """

    def __init__(self, df_config, input_col_name, training_df=None,
                 n_years=3, day_window=1, agg_func='mean', q=None,
                 output_col_suffix='same_doy_lag'):

        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']

        self.input_col_name = input_col_name
        self.n_years = n_years
        self.day_window = day_window
        self.agg_func = agg_func
        self.q = q
        self.output_col_suffix = output_col_suffix

        self.training_df = training_df

    @property
    def training_df(self):
        return self._training_df

    @training_df.setter
    def training_df(self, val):
        self._training_df = val

    def same_day_hour_lag(self, input_df, forecast_creation_time):
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

        day_lag_base = 365
        day_lag_last_year = list(range(day_lag_base - self.day_window,
                                       day_lag_base + self.day_window + 1))
        day_lag_all = []
        for i in range(self.n_years):
            day_lag_all += [j + i * 365 for j in day_lag_last_year]

        day_lag_all = [d for d in day_lag_all if
                        (max_time_stamp - timedelta(days=d)) >= min_time_stamp]

        lag_df = self._create_lag_df(input_df, day_lag_all, frequency='D')
        for col in self.input_col_names:
            day_lag_cols = [c for c in lag_df.columns if c.startwith(col)]
            output_col_name = col + '_' + self.output_col_suffix

            if self.agg_func == 'mean':
                output_df[output_col_name] = round(lag_df[
                    day_lag_cols].mean(
                    axis=1))
            elif self.agg_func == 'quantile' and self.q is not None:
                output_df[output_col_name] = \
                    round(lag_df[day_lag_cols].quantile(self.q, axis=1))
            elif self.agg_func == 'std':
                output_df[output_col_name] = round(lag_df[
                    day_lag_cols].std(
                    axis=1))

        return output_df


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

