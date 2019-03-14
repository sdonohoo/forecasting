from datetime import timedelta
import calendar
import pandas as pd
import numpy as np
import warnings

from sklearn.base import BaseEstimator


class TemporalFeaturizer(BaseEstimator):
    """
    Computes commonly used time-related features.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.

        feature_list(list of str, optional): list of temporal features to
            return. The following features are available:
                hour_of_day
                week_of_year
                month_of_year
                day_of_week
                day_of_month
                day_of_year
                hour_of_year
                week_of_year

            If feature_list is not specified, a default set of features are
            returned depending on df_config['frequency']
            For 'hourly' data: hour_of_day, day_of_week, week_of_year,
            'month_of_year
            For 'daily' data: day_of_week, week_of_year, month_of_year
            For 'weekly' data: week_of_year, month_of_year
            For 'monthly' data: month_of_year
    """

    def __init__(self, df_config, feature_list=None):
        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']

        if feature_list:
            self.feature_list = feature_list
        elif self.frequency == 'hourly':
            self.feature_list = ['hour_of_day', 'day_of_week',
                                 'week_of_year', 'month_of_year']
        elif self.frequency == 'daily':
            self.feature_list = ['day_of_week', 'week_of_year',
                                 'month_of_year']
        elif self.frequency == 'weekly':
            self.feature_list = ['week_of_year', 'month_of_year']
        elif self.frequency == 'monthly':
            self.feature_list = ['month_of_year']
        else:
            raise Exception('Please specify the feature_list, because the '
                            'data frequency is not hourly, daily, weekly, '
                            'or monthly')

        self._feature_function_map = {'hour_of_day': self.hour_of_day,
                                      'week_of_year': self.week_of_year,
                                      'month_of_year': self.month_of_year,
                                      'day_of_week': self.day_of_week,
                                      'day_of_month': self.day_of_month,
                                      'day_of_year': self.day_of_year,
                                      'hour_of_year': self.hour_of_year,
                                      'week_of_month': self.week_of_month}

    def hour_of_day(self, time_col):
        """Returns the hour from a datetime column."""
        return time_col.dt.hour

    def week_of_year(self, time_col):
        """Returns the week from a datetime column."""
        return time_col.dt.week

    def month_of_year(self, time_col):
        """Returns the month from a datetime column."""
        return time_col.dt.month

    def day_of_week(self, time_col):
        """Returns the day of week from a datetime column."""
        return time_col.dt.dayofweek

    def day_of_month(self, time_col):
        """Returns the day of month from a datetime column."""
        return time_col.dt.day

    def day_of_year(self, time_col):
        """Returns the day of year from a datetime column."""
        return time_col.dt.dayofyear

    def week_of_month(self, time_col):
        """Returns the week of month from a datetime column"""
        #Placeholder for week of month feature
        pass

    def hour_of_year(self, time_col):
        """
        Hour of year is a cyclic variable that indicates the annual
        position of a particular hour on a particular day and repeats each
        year. It is each year linearly increasing over time going from 0 on
        January 1 at 00:00 to 1 on December 31st at 23:00. The values
        are normalized to be between [0; 1].

        Args:
            datetime_col: Datetime column.

        Returns:
            A numpy array containing converted the time column to hour of year.
        """

        time_of_year = pd.DataFrame({'day_of_year': time_col.dt.dayofyear,
                                     'hour_of_day': time_col.dt.hour,
                                     'year': time_col.dt.year})
        time_of_year['hour_of_year'] = \
            (time_of_year['day_of_year'] - 1) * 24 + time_of_year['hour_of_day']

        time_of_year['year_length'] = \
            time_of_year['year'].apply(
                lambda y: 366 if calendar.isleap(y) else 365)

        time_of_year['hour_of_year'] = \
            time_of_year['hour_of_year'] / (time_of_year['year_length'] * 24 - 1)

        return time_of_year['hour_of_year'].values

    def fit(self, X, y=None):
        """To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this transformer"""
        return self

    def transform(self, X):
        X = X.copy()
        time_col = X[self.time_col_name]
        for feature in self.feature_list:
            if feature in X.columns:
                warnings.warn('Column {} is already in the data frame, '
                             'it will be overwritten.'.format(feature))
            feature_function = self._feature_function_map[feature]
            X[feature] = feature_function(time_col)

        return X


class DayTypeFeaturizer(BaseEstimator):

    """
    Convert the time column of the input data frame to 7 day types.
    The following mapping is used to convert day of week or holiday to
    integers.
        Monday: 0
        Tuesday, Wednesday, and Thursday: 2
        Friday:4
        Saturday: 5
        Sunday: 6
        Holiday: 7
        Days before and after a holiday: 8

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        holiday_col_name(str, optional): Name of the holiday column. The
            holiday column should contain integers greater than 0 representing
            holidays. The mapping between the holiday and the integer does
            not matter for this featurizer.
        semi_holiday_offset(datetime.timedelta, optional): The time range
            before and after each holiday to be considered as semi-holiday.
            Default value timedelta(days=1).
        weekday_type_map(dict, optional): Mapping multiple weekdays to the same
            number. By default, Tuesday (1) and Thursday (3) are mapped to 2,
            so that Tuesday, Wednesday, and Thursday are treated the same.

        holiday_code(int, optional): Integer used to represent holidays.
            Default value is 7.
        semi_holiday_code(int, optional): Integer used to represent days
            before and after  holiday. Default value is 8

    """
    def __init__(self, df_config, holiday_col_name=None,
                 semi_holiday_offset=timedelta(days=1),
                 weekday_type_map={1: 2, 3: 2},
                 holiday_code=7, semi_holiday_code=8):
        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']

        self.weekday_type_map = weekday_type_map

        self.holiday_col_name = holiday_col_name
        self.semi_holiday_offset = semi_holiday_offset
        self.holiday_code = holiday_code
        self.semi_holiday_code = semi_holiday_code

    def fit(self, X, y=None):
        """To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this transformer"""
        return self

    def transform(self, X):
        X = X.copy()

        datetime_col = X[self.time_col_name]

        datetype = pd.DataFrame({'day_type': datetime_col.dt.dayofweek})
        datetype.replace({'day_type': self.weekday_type_map}, inplace=True)

        if self.holiday_col_name is not None:
            holiday_col = X[self.holiday_col_name].values
            holiday_mask = holiday_col > 0
            datetype.loc[holiday_mask, 'day_type'] = self.holiday_code

            # Create a temporary Date column to calculate dates near
            # the holidays
            datetype['Date'] = pd.to_datetime(datetime_col.dt.date,
                                              format=self.time_format)
            holiday_dates = set(datetype.loc[holiday_mask, 'Date'])

            semi_holiday_dates = \
                [pd.date_range(start=d - self.semi_holiday_offset,
                               end=d + self.semi_holiday_offset,
                               freq='D')
                 for d in holiday_dates]

            # Flatten the list of lists
            semi_holiday_dates = \
                [d for dates in semi_holiday_dates for d in dates]

            semi_holiday_dates = set(semi_holiday_dates)
            semi_holiday_dates = semi_holiday_dates.difference(holiday_dates)

            datetype.loc[datetype['Date'].isin(semi_holiday_dates),
                         'day_type'] \
                = self.semi_holiday_code

        X['day_type'] = datetype['day_type'].values

        return X


def fourier_approximation(t, n, period):
    """
    Generic helper function to create Fourier Series at different
    harmonies (n) and periods.

    Args:
        t(int): Time position, e.g. for daily Fourier series, this is the
            hour of day. For weekly Fourier, this is the day of week. \For
            annual Fourier, this is the day of year.
        n(int): Harmony to compute, n = 1, 2, 3,...
        period(int): Period of the time series data. For hourly data with
            daily seasonality, this is 24. For weekly seasonality, this is 7.
            For yearly seasonality, this is the number of days in a year.

    Returns:
        float: Sine component
        float: Cosine component
    """
    x = n * 2 * np.pi * t / period
    x_sin = np.sin(x)
    x_cos = np.cos(x)

    return x_sin, x_cos


class AnnualFourierFeaturizer(BaseEstimator):
    """
    Creates Annual Fourier Series at different harmonies (n).

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        n_harmonics: Number of harmonies to compute, n=1, 2, 3,...
    """

    def __init__(self, df_config, n_harmonics):
        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']
        self.n_harmonics = n_harmonics

    def fit(self, X, y=None):
        """To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this transformer"""
        return self

    def transform(self, X):
        X = X.copy()
        datetime_col = X[self.time_col_name]
        day_of_year = datetime_col.dt.dayofyear

        output_dict = {}
        for n in range(1, self.n_harmonics + 1):
            sin, cos = fourier_approximation(day_of_year, n, 365.24)

            output_dict['annual_sin_' + str(n)] = sin
            output_dict['annual_cos_' + str(n)] = cos

        for k, v in output_dict.items():
            X[k] = v

        return X


class WeeklyFourierFeaturizer(BaseEstimator):
    """
    Creates Weekly Fourier Series at different harmonies (n).

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        n_harmonics: Number of harmonies to compute, n=1, 2, 3,...

    """
    def __init__(self, df_config, n_harmonics):
        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']
        self.n_harmonics = n_harmonics

    def fit(self, X, y=None):
        """To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this transformer"""
        return self

    def transform(self, X):
        X = X.copy()
        datetime_col = X[self.time_col_name]
        day_of_week = datetime_col.dt.dayofweek + 1

        output_dict = {}
        for n in range(1, self.n_harmonics + 1):
            sin, cos = fourier_approximation(day_of_week, n, 7)

            output_dict['weekly_sin_' + str(n)] = sin
            output_dict['weekly_cos_' + str(n)] = cos

        for k, v in output_dict.items():
            X[k] = v

        return X


class DailyFourierFeaturizer(BaseEstimator):
    """
    Creates Daily Fourier Series at different harmonies (n).

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        n_harmonics: Number of harmonies to compute, n=1, 2, 3,...

    """
    def __init__(self, df_config, n_harmonics):
        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']
        self.n_harmonics = n_harmonics

    def fit(self, X, y=None):
        """To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this transformer"""
        return self

    def transform(self, X):
        X = X.copy()
        datetime_col = X[self.time_col_name]
        hour_of_day = datetime_col.dt.hour + 1

        output_dict = {}
        for n in range(1, self.n_harmonics + 1):
            sin, cos = fourier_approximation(hour_of_day, n, 24)

            output_dict['daily_sin_' + str(n)] = sin
            output_dict['daily_cos_' + str(n)] = cos

        for k, v in output_dict.items():
            X[k] = v

        return X



