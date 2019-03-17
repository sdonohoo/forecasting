#TODO: Moving features for retail benchmark

import pandas as pd
from sklearn.base import BaseEstimator
from ..utils import is_datetime_like


class SameWeekdayHourRollingAggFeaturizer(BaseEstimator):
    """
    Creates a series of rolling window aggregation features.

    These features are useful for data with both weekly and daily
    seasonalities and has hourly frequency. The features are computed by
    calculating the mean, quantiles or std of values of the same day of week
    and same hour of day of recent weeks.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        input_col_name(str): Name of the column to create the lag feature on.
        start_week(int): Number of weeks between the current week and the first
            week of the first aggregation feature.
        window_size(int): Number of weeks used to compute each aggregation.
            The window is on the "older" side of the first week of each window.
            For example, if start_week is 9 and window_size is 4, the output
            feature moving_agg_lag_9 aggregates the same day and hour values
            of the 9th, 10th, 11th, and 12th weeks before the current week.
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
        max_test_timestamp(pd.datetime): Maximum timestamp of the testing
            data to generate forecasting on. This value is needed to prevent
            creating lag features on the training data that are not available
            for the testing data. For example, the features and models are
            created on week 7 to forecast week 8 to week 10. It would not make
            sense to create an aggregation feature using data from week 8 and
            week 9, because they are not available at the forecast creation
            time. Thus, it does not make sense to create an aggregation
            feature using data from week 5 and week 6 for week 7.
        agg_count(int): Number of aggregation features to create. Default
            value is 1.
            For example, start_week = 9, window_size=4, and count = 3 will
            create three aggregation of features.
            1) moving_agg_lag_9: aggregate the same day and hour values of
            the 9th, 10th, 11th, and 12th weeks before the current week.
            2) moving_agg_lag_10: aggregate the same day and hour values of the
            10th, 11th, 12th, and 13th weeks before the current week.
            3) moving_agg_lag_11: aggregate the same day and hour values of the
            11th, 12th, 13th, and 14th weeks before the current week.
        agg_func(str): Aggregation function to apply on multiple previous
            values, accepted values are 'mean', 'quantile', 'std'. Default
            value is 'mean'.
        q(float): Quantile to compute from previous values, if agg_func is
            'quantile', taking value between 0 and 1.
        output_col_prefix(str): Prefix of the output columns. The start week of
            each moving average feature is added at the end. Default value
            'rolling_agg_lag_'.
    """

    def __init__(self, df_config, input_col_name,
                 start_week, window_size,
                 max_test_timestamp, training_df=None,
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

        self.training_df = training_df
        self.max_test_timestamp = max_test_timestamp

    @property
    def training_df(self):
        return self._training_df

    @training_df.setter
    def training_df(self, val):
        self._training_df = val

    def same_weekday_hour_rolling_agg(self, input_df, forecast_creation_time):
        datetime_col = input_df[self.time_col_name]
        input_col = input_df[self.input_col_name]

        if not is_datetime_like(datetime_col):
            datetime_col = pd.to_datetime(datetime_col, format=self.time_format)

        # Create a temporary data frame to perform shifting on
        df = pd.DataFrame({'Datetime': datetime_col, 'value': input_col})
        df.set_index('Datetime', inplace=True)

        # TODO: Change this to self.frequency after updating data schema
        df = df.asfreq('H')

        if not df.index.is_monotonic:
            df.sort_index(inplace=True)

        # Compute the difference in hours between the maximum timestamp and the
        # forecast creation time. Any shift smaller than this is not allowed
        # because it's not available for all forecasting time points.
        max_fct_diff = df.index.max() - forecast_creation_time
        max_fct_diff = max_fct_diff.days*24 + max_fct_diff.seconds/3600

        for i in range(self.agg_count):
            output_col = self.output_col_prefix + str(self.start_week + i)
            week_lag_start = self.start_week + i
            hour_lags = \
                [(week_lag_start + w) * 24 * 7 for w in range(self.window_size)]
            hour_lags = [h for h in hour_lags if h > max_fct_diff]
            if len(hour_lags) > 0:
                tmp_df = df[['value']].copy()
                tmp_col_all = []
                for h in hour_lags:
                    tmp_col = 'tmp_lag_' + str(h)
                    tmp_col_all.append(tmp_col)
                    tmp_df[tmp_col] = tmp_df['value'].shift(h)

                if self.agg_func == 'mean':
                    df[output_col] = round(tmp_df[tmp_col_all].mean(axis=1))
                elif self.agg_func == 'quantile' and self.q is not None:
                    df[output_col] = \
                        round(tmp_df[tmp_col_all].quantile(self.q, axis=1))
                elif self.agg_func == 'std':
                    df[output_col] = round(tmp_df[tmp_col_all].std(axis=1))

        df.drop(['value'], inplace=True, axis=1)

        return df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(self.grain_col_name, list):
            col_names = [self.time_col_name, self.input_col_name] + \
                        self.grain_col_name
            merge_col_names = [self.time_col_name] + self.grain_col_name
        else:
            col_names = [self.time_col_name, self.input_col_name,
                         self.grain_col_name]
            merge_col_names = [self.time_col_name, self.grain_col_name]

        if self.training_df is not None:
            forecast_creation_time = self.training_df[self.time_col_name].max()
            X_tmp = pd.concat([self.training_df, X], sort=True)
            X_tmp = X_tmp[col_names].copy()
        else:
            # Compute an imaginary forecast creation time for the training
            # data based on the maximum timestamp to forecast on
            max_train_timestamp = X[self.time_col_name].max()
            train_test_timestamp_diff = \
                self.max_test_timestamp - max_train_timestamp
            forecast_creation_time = \
                max_train_timestamp - train_test_timestamp_diff

            X_tmp = X[col_names].copy()

        if self.grain_col_name is None:
            output_tmp = \
                self.same_weekday_hour_rolling_agg(X_tmp,
                                                   forecast_creation_time)
            if self.training_df is not None:
                output_tmp = output_tmp.loc[output_tmp[self.time_col_name] >
                                            forecast_creation_time].copy()
            X = pd.merge(X, output_tmp, on=self.time_col_name)
        else:
            output_tmp = \
                X_tmp.groupby(self.grain_col_name).apply(
                    lambda g: self.same_weekday_hour_rolling_agg(
                        g, forecast_creation_time))
            output_tmp.reset_index(inplace=True)

            if self.training_df is not None:
                output_tmp = output_tmp.loc[output_tmp[self.time_col_name] >
                                            forecast_creation_time].copy()

            X = pd.merge(X, output_tmp, on=merge_col_names)
        if X.shape[0] == 0:
            raise Exception('The featurizer output is empty. Set the '
                            'training_df property of the featurizer to '
                            'None if transforming training data.')
        return X

