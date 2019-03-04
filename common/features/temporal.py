from datetime import timedelta
import calendar
import pandas as pd
import numpy as np
import warnings

from sklearn.base import BaseEstimator


class TemporalFeaturizer(BaseEstimator):

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
                                 'day_of_month', 'day_of_year', 'week_of_year',
                                 'month_of_year']
        elif self.frequency == 'daily':
            self.feature_list = ['day_of_week',
                                 'day_of_month', 'day_of_year', 'week_of_year',
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
                                      'day_of_year': self.day_of_year}

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
        pass

    def hour_of_year(self, time_col):
        """
        Time of year is a cyclic variable that indicates the annual position and
        repeats each year. It is each year linearly increasing over time going
        from 0 on January 1 at 00:00 to 1 on December 31st at 23:00. The values
        are normalized to be between [0; 1].

        Args:
            datetime_col: Datetime column.

        Returns:
            A numpy array containing converted datatime_col into time of year.
        """

        time_of_year = pd.DataFrame({'DayOfYear': time_col.dt.dayofyear,
                                     'HourOfDay': time_col.dt.hour,
                                     'Year': time_col.dt.year})
        time_of_year['TimeOfYear'] = \
            (time_of_year['DayOfYear'] - 1) * 24 + time_of_year['HourOfDay']

        time_of_year['YearLength'] = \
            time_of_year['Year'].apply(
                lambda y: 366 if calendar.isleap(y) else 365)

        time_of_year['TimeOfYear'] = \
            time_of_year['TimeOfYear'] / (time_of_year['YearLength'] * 24 - 1)

        return time_of_year['TimeOfYear'].values

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        time_col = X[self.time_col_name]
        for feature in self.feature_list:
            if feature in X.columns:
                warnings.warn('Column {} is already in the data frame, '
                             'it will be overwritten.'.format(feature))
            feature_function = self._feature_function_map[feature]
            X[feature] = feature_function(time_col)

        return X


class DateTypeFeaturizer(BaseEstimator):

    """
    Convert datetime_col to 7 day types
    0: Monday
    2: Tuesday, Wednesday, and Thursday
    4: Friday
    5: Saturday
    6: Sunday
    7: Holiday
    8: Days before and after a holiday

    Args:
        datetime_col: Datetime column.
        holiday_col: Holiday code column. Default value None.
        semi_holiday_offset: Time difference between the date before (or after)
            the holiday and the holiday. Default value timedelta(days=1).

    Returns:
        A numpy array containing converted datatime_col into day types.
    """
    def __init__(self, df_config, holiday_col_name=None,
                 semi_holiday_offset=timedelta(days=1),
                 weekday_type_map={1: 2, 3: 2},
                 holiday_code=7, semi_holiday_code = 8):
        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']

        self.weekday_type_map = weekday_type_map

        self.holiday_col_name = holiday_col_name
        self.semi_holiday_offset = semi_holiday_offset
        self.holidday_code = holiday_code
        self.semi_holiday_code = semi_holiday_code

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        datetime_col = X[self.time_col_name]

        datetype = pd.DataFrame({'DayType': datetime_col.dt.dayofweek})
        datetype.replace({'DayType': self.weekday_type_map}, inplace=True)

        if self.holiday_col_name is not None:
            holiday_col = X[self.holiday_col_name]
            holiday_mask = holiday_col > 0
            datetype.loc[holiday_mask, 'DayType'] = self.holidday_code

            # Create a temporary Date column to calculate dates near the holidays
            datetype['Date'] = pd.to_datetime(datetime_col.dt.date,
                                              format=self.time_format)
            holiday_dates = set(datetype.loc[holiday_mask, 'Date'])

            semi_holiday_dates = \
                [pd.date_range(start=d - self.semi_holiday_offset,
                               end=d + self.semi_holiday_offset,
                               freq='D')
                 for d in holiday_dates]

            # Flatten the list of lists
            semi_holiday_dates = [d for dates in semi_holiday_dates for d in dates]

            semi_holiday_dates = set(semi_holiday_dates)
            semi_holiday_dates = semi_holiday_dates.difference(holiday_dates)

            datetype.loc[datetype['Date'].isin(semi_holiday_dates), 'DayType'] \
                = self.semi_holiday_code

        X['DayType'] = datetype['DayType'].values

        return X


def fourier_approximation(t, n, period):
    """
    Generic helper function to create Fourier Series at different harmonies (n) and periods.

    Args:
        t: Datetime column.
        n: Harmonies, n=0, 1, 2, 3,...
        period: Period of the datetime variable t.

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
        datetime_col: Datetime column.
        n_harmonics: Harmonies, n=0, 1, 2, 3,...

    Returns:
        dict: Output dictionary containing sine and cosine components of
            the Fourier series for all harmonies.
    """

    def __init__(self, df_config, n_harmonics):
        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']
        self.n_harmonics = n_harmonics

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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
        datetime_col: Datetime column.
        n_harmonics: Harmonies, n=0, 1, 2, 3,...

    Returns:
        dict: Output dictionary containing sine and cosine components of
            the Fourier series for all harmonies.
    """
    def __init__(self, df_config, n_harmonics):
        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']
        self.n_harmonics = n_harmonics

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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
        datetime_col: Datetime column.
        n_harmonics: Harmonies, n=0, 1, 2, 3,...

    Returns:
        dict: Output dictionary containing sine and cosine components of
            the Fourier series for all harmonies.
    """
    def __init__(self, df_config, n_harmonics):
        self.time_col_name = df_config['time_col_name']
        self.value_col_name = df_config['value_col_name']
        self.grain_col_name = df_config['grain_col_name']
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']
        self.n_harmonics = n_harmonics

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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



