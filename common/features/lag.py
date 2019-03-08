from datetime import timedelta
import pandas as pd
import numpy as np

from ..utils import is_datetime_like

from sklearn.base import BaseEstimator


class LagFeaturizer():
    # Placeholder for multiple Lag features with a gap
    pass


class SameWeekDayHourLagFeaturizer(BaseEstimator):
    """
    Creates a lag feature by calculating quantiles, mean and std of values of and
    around the same week, same day of week, and same hour of day, of previous years.

    Args:
        datetime_col: Datetime column.
        value_col: Feature value column to create lag feature from.
        n_years: Number of previous years data to use. Default value 3.
        week_window: Number of weeks before and after the same week to use,
            which should help reduce noise in the data. Default value 1.
        agg_func: Aggregation function to apply on multiple previous values,
            accepted values are 'mean', 'quantile', 'std'. Default value 'mean'.
        q: If agg_func is 'quantile', taking value between 0 and 1.
        output_colname: name of the output lag feature column.
            Default value 'SameWeekHourLag'.

    Returns:
         pandas.DataFrame: data frame containing the newly created lag
            feature as a column.
    """

    def __init__(self, df_config, input_col_name, training_df=None, n_years=3,
                 week_window=1, agg_func='mean', q=None,
                 output_col_name='SameWeekDayHourLag'):
        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']

        self.input_col_name = input_col_name
        self.n_years = n_years
        self.week_window = week_window
        self.agg_func = agg_func
        self.q = q
        self.output_col_name = output_col_name

        self._training_df = training_df

    @property
    def training_df(self):
        return self._training_df

    @training_df.setter
    def training_df(self, val):
        self._training_df = val

    def same_weekday_hour_lag(self, input_df):
        datetime_col = input_df[self.time_col_name]
        input_col = input_df[self.input_col_name]
        if not is_datetime_like(datetime_col):
            datetime_col = pd.to_datetime(datetime_col, format=self.time_format)
        min_time_stamp = min(datetime_col)
        max_time_stamp = max(datetime_col)

        df = pd.DataFrame({'Datetime': datetime_col, 'value': input_col})
        df.set_index('Datetime', inplace=True)

        week_lag_base = 52
        week_lag_last_year = list(range(week_lag_base - self.week_window,
                                        week_lag_base + self.week_window + 1))
        week_lag_all = []
        for y in range(self.n_years):
            week_lag_all += [x + y * 52 for x in week_lag_last_year]

        week_lag_cols = []
        for w in week_lag_all:
            if (max_time_stamp - timedelta(weeks=w)) >= min_time_stamp:
                col_name = 'week_lag_' + str(w)
                week_lag_cols.append(col_name)

                lag_datetime = df.index.get_level_values(0) - timedelta(weeks=w)
                valid_lag_mask = lag_datetime >= min_time_stamp

                df[col_name] = np.nan

                df.loc[valid_lag_mask, col_name] = \
                    df.loc[lag_datetime[valid_lag_mask], 'value'].values

        if self.agg_func == 'mean' and self.q is None:
            df[self.output_col_name] = round(df[week_lag_cols].mean(axis=1))
        elif self.agg_func == 'quantile' and self.q is not None:
            df[self.output_col_name] = \
                round(df[week_lag_cols].quantile(self.q, axis=1))
        elif self.agg_func == 'std' and self.q is None:
            df[self.output_col_name] = round(df[week_lag_cols].std(axis=1))

        return df[[self.output_col_name]]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.training_df is not None:
            forecast_creation_time = max(self.training_df[self.time_col_name])
            X = pd.concat([self.training_df, X])
        else:
            forecast_creation_time = max(X[self.time_col_name])
            X = X.copy()
        if self.grain_col_name is None:
            X[self.output_col_name] = \
                self.same_weekday_hour_lag(X).values
            if self.training_df is not None:
                X = X.loc[X[self.time_col_name] > forecast_creation_time].copy()
        else:
            if isinstance(self.grain_col_name, list):
                col_names = [self.time_col_name, self.input_col_name] + \
                            self.grain_col_name
                merge_col_names = [self.time_col_name] + self.grain_col_name
            else:
                col_names = [self.time_col_name, self.input_col_name,
                             self.grain_col_name]
                merge_col_names = [self.time_col_name, self.grain_col_name]
            X_lag_tmp = \
                X[col_names].groupby(self.grain_col_name)\
                    .apply(lambda g: self.same_weekday_hour_lag(g))
            X_lag_tmp.reset_index(inplace=True)

            if self.training_df is not None:
                X_lag_tmp = X_lag_tmp.loc[X_lag_tmp[self.time_col_name] >
                                          forecast_creation_time].copy()
            X = pd.merge(X, X_lag_tmp, on=merge_col_names)

        return X


class SameDayHourLagFeaturizer(BaseEstimator):

    """
    Creates a lag feature by calculating quantiles, mean, and std of values of
    and around the same day of year, and same hour of day, of previous years.

    Args:
        datetime_col: Datetime column.
        value_col: Feature value column to create lag feature from.
        n_years: Number of previous years data to use. Default value 3.
        day_window: Number of days before and after the same day to use,
            which should help reduce noise in the data. Default value 1.
        agg_func: Aggregation function to apply on multiple previous values,
            accepted values are 'mean', 'quantile', 'std'. Default value 'mean'.
        q: If agg_func is 'quantile', taking value between 0 and 1.
        output_colname: name of the output lag feature column.
            Default value 'SameDayHourLag'.

    Returns:
        pandas.DataFrame: data frame containing the newly created lag
            feature as a column.
    """

    def __init__(self, df_config, input_col_name, training_df=None,
                 n_years=3, day_window=1, agg_func='mean', q=None,
                 output_col_name='SameDayHourLag'):

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
        self.output_col_name = output_col_name

        self.training_df = training_df

    @property
    def training_df(self):
        return self._training_df

    @training_df.setter
    def training_df(self, val):
        self._training_df = val

    def same_day_hour_lag(self, input_df):
        datetime_col = input_df[self.time_col_name]
        input_col = input_df[self.input_col_name]
        if not is_datetime_like(datetime_col):
            datetime_col = pd.to_datetime(datetime_col, format=self.time_format)
        min_time_stamp = min(datetime_col)
        max_time_stamp = max(datetime_col)

        df = pd.DataFrame({'Datetime': datetime_col, 'value': input_col})
        df.set_index('Datetime', inplace=True)

        day_lag_base = 365
        day_lag_last_year = list(range(day_lag_base - self.day_window,
                                       day_lag_base + self.day_window + 1))
        day_lag_all = []
        for y in range(self.n_years):
            day_lag_all += [x + y * 365 for x in day_lag_last_year]

        day_lag_cols = []
        for d in day_lag_all:
            if (max_time_stamp - timedelta(days=d)) >= min_time_stamp:
                col_name = 'day_lag_' + str(d)
                day_lag_cols.append(col_name)

                lag_datetime = df.index.get_level_values(0) - timedelta(days=d)
                valid_lag_mask = lag_datetime >= min_time_stamp

                df[col_name] = np.nan

                df.loc[valid_lag_mask, col_name] = \
                    df.loc[lag_datetime[valid_lag_mask], 'value'].values

        # Additional aggregation options will be added as needed
        if self.agg_func == 'mean' and self.q is None:
            df[self.output_col_name] = round(df[day_lag_cols].mean(axis=1))
        elif self.agg_func == 'quantile' and self.q is not None:
            df[self.output_col_name] = \
                round(df[day_lag_cols].quantile(self.q, axis=1))
        elif self.agg_func == 'std' and self.q is None:
            df[self.output_col_name] = round(df[day_lag_cols].std(axis=1))

        return df[[self.output_col_name]]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.training_df is not None:
            forecast_creation_time = max(self.training_df[self.time_col_name])
            X = pd.concat([self.training_df, X])
        else:
            forecast_creation_time = max(X[self.time_col_name])
            X = X.copy()

        if self.grain_col_name is None:
            X[self.output_col_name] = \
                self.same_day_hour_lag(X).values
            if self.training_df is not None:
                X = X.loc[X[self.time_col_name] > forecast_creation_time].copy()
        else:
            if isinstance(self.grain_col_name, list):
                col_names = [self.time_col_name, self.input_col_name] + \
                            self.grain_col_name
                merge_col_names = [self.time_col_name] + self.grain_col_name
            else:
                col_names = [self.time_col_name, self.input_col_name,
                             self.grain_col_name]
                merge_col_names = [self.time_col_name, self.grain_col_name]
            X_lag_tmp = \
                X[col_names].groupby(self.grain_col_name)\
                    .apply(lambda g: self.same_day_hour_lag(g))
            X_lag_tmp.reset_index(inplace=True)

            if self.training_df is not None:
                X_lag_tmp = X_lag_tmp.loc[X_lag_tmp[self.time_col_name] >
                                          forecast_creation_time].copy()

            X = pd.merge(X, X_lag_tmp, on=merge_col_names)

        return X




