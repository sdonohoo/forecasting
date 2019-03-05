#TODO: Moving features for retail benchmark

import pandas as pd
from functools import reduce
from sklearn.base import BaseEstimator
from .lag import SameWeekDayHourLagFeaturizer


class SameWeekDayHourRollingFeaturizer(BaseEstimator):
    def __init__(self, df_config, input_col_name, window_size, start_week,
                 agg_count=1, agg_func='mean', q=None,
                 output_col_prefix='rolling_agg_lag_'):
        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']

        self.input_col_name = input_col_name
        self.window_size = window_size
        self.start_week = start_week
        self.agg_count = agg_count
        self.agg_func = agg_func
        self.q = q
        self.output_col_prefix = output_col_prefix

        self._is_fit = False

    def same_weekday_hour_rolling_agg(self, input_df):
        """
        Creates a series of aggregation features by calculating mean, quantiles,
        or std of values of the same day of week and same hour of day of previous weeks.

        Args:
            datetime_col: Datetime column
            value_col: Feature value column to create aggregation features from.
            window_size: Number of weeks used to compute the aggregation.
            start_week: First week of the first aggregation feature.
            count: Number of aggregation features to create.
            forecast_creation_time: The time point when the feature is created.
                This value is used to prevent using data that are not available
                at forecast creation time to compute features.
            agg_func: Aggregation function to apply on multiple previous values,
                accepted values are 'mean', 'quantile', 'std'.
            q: If agg_func is 'quantile', taking value between 0 and 1.
            output_col_prefix: Prefix of the output columns. The start week of each
                moving average feature is added at the end. Default value 'moving_agg_lag_'.

        Returns:
            pandas.DataFrame: data frame containing the newly created lag features as
                columns.

        For example, start_week = 9, window_size=4, and count = 3 will
        create three aggregation of features.
        1) moving_agg_lag_9: aggregate the same day and hour values of the 9th,
        10th, 11th, and 12th weeks before the current week.
        2) moving_agg_lag_10: aggregate the same day and hour values of the
        10th, 11th, 12th, and 13th weeks before the current week.
        3) moving_agg_lag_11: aggregate the same day and hour values of the
        11th, 12th, 13th, and 14th weeks before the current week.
        """
        datetime_col = input_df[self.time_col_name]
        input_col = input_df[self.input_col_name]

        df = pd.DataFrame({'Datetime': datetime_col, 'value': input_col})
        df.set_index('Datetime', inplace=True)

        df = df.asfreq('H')

        if not df.index.is_monotonic:
            df.sort_index(inplace=True)

        df['fct_diff'] = df.index - self.forecast_creation_time
        df['fct_diff'] = df['fct_diff'].apply(
            lambda x: x.days * 24 + x.seconds / 3600)
        max_diff = max(df['fct_diff'])

        for i in range(self.agg_count):
            output_col = self.output_col_prefix + str(self.start_week + i)
            week_lag_start = self.start_week + i
            hour_lags = \
                [(week_lag_start + w) * 24 * 7 for w in range(self.window_size)]
            hour_lags = [h for h in hour_lags if h > max_diff]
            if len(hour_lags) > 0:
                tmp_df = df[['value']].copy()
                tmp_col_all = []
                for h in hour_lags:
                    tmp_col = 'tmp_lag_' + str(h)
                    tmp_col_all.append(tmp_col)
                    tmp_df[tmp_col] = tmp_df['value'].shift(h)

            if self.agg_func == 'mean' and self.q is None:
                df[output_col] = round(tmp_df[tmp_col_all].mean(axis=1))
            elif self.agg_func == 'quantile' and self.q is not None:
                df[output_col] = round(tmp_df[tmp_col_all].quantile(q, axis=1))
            elif self.agg_func == 'std' and self.q is None:
                df[output_col] = round(tmp_df[tmp_col_all].std(axis=1))

        df.drop(['fct_diff', 'value'], inplace=True, axis=1)

        return df

    def fit(self, X, y=None):
        self.forecast_creation_time = max(X[self.time_col_name])
        self._is_fit = True
        return self

    def transform(self, X):
        ## TODO: raise an exception when the transformer is not fit
        if self.grain_col_name is None:
            output_tmp = self.same_weekday_hour_rolling_agg(X)
            X = pd.merge(X, output_tmp, on=self.time_col_name)
        else:
            ##TODO: Need to handle when grain column name is a list
            output_tmp = \
                X[[self.time_col_name, self.input_col_name,
                   self.grain_col_name]].groupby(self.grain_col_name)\
                    .apply(lambda g: self.same_weekday_hour_rolling_agg(g))
            output_tmp.reset_index(inplace=True)
            X = pd.merge(X, output_tmp,
                         on=[self.grain_col_name, self.time_col_name])

        return X


class YearOverYearRatioFeaturizer(BaseEstimator):

    def __init__(self, df_config, input_col_name,
                 n_years, column_prefix, output_col_prefix):
        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']
        self.df_config = df_config

        self.input_col_name = input_col_name
        self.n_years = n_years
        self.column_prefix = column_prefix
        self.output_col_prefix = output_col_prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        column_prefix_new = self.column_prefix + 'lag_'
        lag_df_list = []
        columns_old = [col for col in X if col.startswith(self.column_prefix)]
        for col_old in columns_old:
            col_new = col_old.replace(self.column_prefix, column_prefix_new)
            col_ratio = col_old.replace(self.column_prefix,
                                        self.output_col_prefix)
            same_weekday_hour_lag_featurizer = \
                SameWeekDayHourLagFeaturizer(df_config=self.df_config,
                                             input_col_name=col_old,
                                             n_years=self.n_years,
                                             week_window=0,
                                             output_col_name=col_new)

            lag_df = same_weekday_hour_lag_featurizer.transform(X)

            lag_df.reset_index(inplace=True)
            lag_df[col_ratio] = X[col_old] / lag_df[col_new]
            lag_df.drop(col_new, inplace=True, axis=1)
            lag_df_list.append(lag_df)

        ##TODO: Need to handle when grain column name is a list
        output_df = reduce(
            lambda left, right: pd.merge(left, right,
                                         on=[self.time_col_name,
                                             self.grain_col_anme]),
            [X] + lag_df_list)

        return output_df