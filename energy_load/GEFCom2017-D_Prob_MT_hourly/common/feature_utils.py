"""
This file contains helper functions for creating features on the GEFCom2017-D
dataset.
"""

from datetime import timedelta
import calendar
import pandas as pd
import numpy as np
from functools import reduce

from utils import is_datetime_like

# 0: Monday, 2: T/W/TR, 4: F, 5:SA, 6: S
WEEK_DAY_TYPE_MAP = {1: 2, 3: 2}    # Map for converting Wednesday and
                                    # Thursday to have the same code as Tuesday
HOLIDAY_CODE = 7
SEMI_HOLIDAY_CODE = 8  # days before and after a holiday

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

def get_datetime_col(df, datetime_colname):
    """
    Helper function for extracting the datetime column as datetime type from
    a data frame.
    """
    if datetime_colname in df.index.names:
        datetime_col = df.index.get_level_values(datetime_colname)
    elif datetime_colname in df.columns:
        datetime_col = df[datetime_colname]
    else:
        raise Exception('Column or index {0} does not exist in the data '
                        'frame'.format(datetime_colname))

    if not is_datetime_like(datetime_col):
        try:
            datetime_col = pd.to_datetime(df[datetime_colname],
                                          format=DATETIME_FORMAT)
        except:
            raise Exception('Column or index {0} can not be converted to '
                            'datetime type.'.format(datetime_colname))
    return datetime_col


def day_type(datetime_col, holiday_col=None,
             semi_holiday_offset=timedelta(days=1)):
    """
    Convert datetime_col to 7 day types
    0: Monday
    2: Tuesday, Wednesday, and Thursday
    4: Friday
    5: Saturday
    6: Sunday
    7: Holiday
    8: Days before and after a holiday
    """
    datetype = pd.DataFrame({'DayType': datetime_col.dt.dayofweek})
    datetype.replace({'DayType': WEEK_DAY_TYPE_MAP}, inplace=True)

    if holiday_col is not None:
        holiday_mask = holiday_col > 0
        datetype.loc[holiday_mask, 'DayType'] = HOLIDAY_CODE

        # Create a temporary Date column to calculate dates near the holidays
        datetype['Date'] = pd.to_datetime(datetime_col.dt.date,
                                          format=DATETIME_FORMAT)
        holiday_dates = set(datetype.loc[holiday_mask, 'Date'])

        semi_holiday_dates = \
            [pd.date_range(start=d - semi_holiday_offset,
                           end=d + semi_holiday_offset,
                           freq='D')
             for d in holiday_dates]

        # Flatten the list of lists
        semi_holiday_dates = [d for dates in semi_holiday_dates for d in dates]

        semi_holiday_dates = set(semi_holiday_dates)
        semi_holiday_dates = semi_holiday_dates.difference(holiday_dates)

        datetype.loc[datetype['Date'].isin(semi_holiday_dates), 'DayType'] \
            = SEMI_HOLIDAY_CODE

    return datetype['DayType'].values


def hour_of_day(datetime_col):
    return datetime_col.dt.hour


def time_of_year(datetime_col):
    """
    Time of year is a cyclic variable that indicates the annual position and
    repeats each year. It is each year linearly increasing over time going
    from 0 on January 1 at 00:00 to 1 on December 31st at 23:00. The values
    are normalized to be between [0; 1].
    """
    time_of_year = pd.DataFrame({'DayOfYear': datetime_col.dt.dayofyear,
                                 'HourOfDay': datetime_col.dt.hour,
                                 'Year': datetime_col.dt.year})
    time_of_year['TimeOfYear'] = \
        (time_of_year['DayOfYear'] - 1) * 24 + time_of_year['HourOfDay']

    time_of_year['YearLength'] = \
        time_of_year['Year'].apply(
            lambda y: 366 if calendar.isleap(y) else 365)

    time_of_year['TimeOfYear'] = \
        time_of_year['TimeOfYear']/(time_of_year['YearLength'] * 24 - 1)

    return time_of_year['TimeOfYear'].values


def week_of_year(datetime_col):
    return datetime_col.dt.week


def normalized_current_year(datetime_col, min_year, max_year):

    year = datetime_col.dt.year
    current_year = (year - min_year)/(max_year - min_year)

    return current_year


def fourier_approximation(t, n, period):
    x = n * 2 * np.pi * t/period
    x_sin = np.sin(x)
    x_cos = np.cos(x)

    return x_sin, x_cos


def annual_fourier(datetime_col, n_harmonics):
    day_of_year = datetime_col.dt.dayofyear

    output_dict = {}
    for n in range(1, n_harmonics+1):
        sin, cos = fourier_approximation(day_of_year, n, 365.24)

        output_dict['annual_sin_'+str(n)] = sin
        output_dict['annual_cos_'+str(n)] = cos

    return output_dict


def weekly_fourier(datetime_col, n_harmonics):
    day_of_week = datetime_col.dt.dayofweek

    output_dict = {}
    for n in range(1, n_harmonics+1):
        sin, cos = fourier_approximation(day_of_week, n, 7)

        output_dict['weekly_sin_'+str(n)] = sin
        output_dict['weekly_cos_'+str(n)] = cos

    return output_dict


def daily_fourier(datetime_col, n_harmonics):
    hour_of_day = datetime_col.dt.hour + 1

    output_dict = {}
    for n in range(1, n_harmonics+1):
        sin, cos = fourier_approximation(hour_of_day, n, 24)

        output_dict['daily_sin_'+str(n)] = sin
        output_dict['daily_cos_'+str(n)] = cos

    return output_dict


def same_week_day_hour_lag(datetime_col, value_col, n_years=3,
                           week_window=1, agg_func='mean',
                           output_colname='SameWeekHourLag'):
    """
    Create a lag feature by averaging values of and around the same week,
    same day of week, and same hour of day, of previous years.
    :param datetime_col: Datetime column
    :param value_col: Feature value column to create lag feature from
    :param n_years: Number of previous years data to use
    :param week_window:
        Number of weeks before and after the same week to
        use, which should help reduce noise in the data
    :param agg_func: aggregation function to apply on multiple previous values
    :param output_colname: name of the output lag feature column
    """

    if not is_datetime_like(datetime_col):
        datetime_col = pd.to_datetime(datetime_col, format=DATETIME_FORMAT)
    min_time_stamp = min(datetime_col)
    max_time_stamp = max(datetime_col)

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    week_lag_base = 52
    week_lag_last_year = list(range(week_lag_base - week_window,
                              week_lag_base + week_window + 1))
    week_lag_all = []
    for y in range(n_years):
        week_lag_all += [x + y * 52 for x in week_lag_last_year]

    week_lag_cols = []
    for w in week_lag_all:
        if max_time_stamp - timedelta(weeks=w) >= min_time_stamp:
            col_name = 'week_lag_' + str(w)
            week_lag_cols.append(col_name)

            lag_datetime = df.index.get_level_values(0) - timedelta(weeks=w)
            valid_lag_mask = lag_datetime >= min_time_stamp

            df[col_name] = np.nan

            df.loc[valid_lag_mask, col_name] = \
                df.loc[lag_datetime[valid_lag_mask], 'value'].values

    # Additional aggregation options will be added as needed
    if agg_func == 'mean':
        df[output_colname] = round(df[week_lag_cols].mean(axis=1))

    return df[[output_colname]]


def same_day_hour_lag(datetime_col, value_col, n_years=3,
                      day_window=1, agg_func='mean',
                      output_colname='SameDayHourLag'):
    """
    Create a lag feature by averaging values of and around the same day of
    year, and same hour of day, of previous years.
    :param datetime_col: Datetime column
    :param value_col: Feature value column to create lag feature from
    :param n_years: Number of previous years data to use
    :param day_window:
        Number of days before and after the same day to
        use, which should help reduce noise in the data
    :param agg_func: aggregation function to apply on multiple previous values
    :param output_colname: name of the output lag feature column
    """

    if not is_datetime_like(datetime_col):
        datetime_col = pd.to_datetime(datetime_col, format=DATETIME_FORMAT)
    min_time_stamp = min(datetime_col)
    max_time_stamp = max(datetime_col)

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    day_lag_base = 365
    day_lag_last_year = list(range(day_lag_base - day_window,
                                   day_lag_base + day_window + 1))
    day_lag_all = []
    for y in range(n_years):
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
    if agg_func == 'mean':
        df[output_colname] = round(df[day_lag_cols].mean(axis=1))

    return df[[output_colname]]


def create_features(input_df, datetime_colname,
                    holiday_colname=None, one_hot_encode=False):

    categorical_columns = ['DayType', 'Hour', 'WeekOfYear']

    output_df = input_df.copy()
    if not is_datetime_like(output_df[datetime_colname]):
        output_df[datetime_colname] = \
            pd.to_datetime(output_df[datetime_colname], format=DATETIME_FORMAT)
    datetime_col = output_df[datetime_colname]

    # Basic temporal features
    output_df['DayType'] = day_type(datetime_col, output_df[holiday_colname])
    output_df['Hour'] = hour_of_day(datetime_col)
    output_df['TimeOfYear'] = time_of_year(datetime_col)
    output_df['WeekOfYear'] = week_of_year(datetime_col)
    output_df['CurrentYear'] = normalized_current_year(
        datetime_col, 2011, 2017)

    # Fourier approximation features
    annual_fourier_approx = annual_fourier(datetime_col, n_harmonics=3)
    weekly_fourier_approx = weekly_fourier(datetime_col, n_harmonics=3)
    daily_fourier_approx = daily_fourier(datetime_col, n_harmonics=2)

    for k, v in annual_fourier_approx.items():
        output_df[k] = v

    for k, v in weekly_fourier_approx.items():
        output_df[k] = v

    for k, v in daily_fourier_approx.items():
        output_df[k] = v

    # Load lag
    same_week_day_hour_load_lag = \
        output_df[[datetime_colname, 'DEMAND', 'Zone']].groupby('Zone').apply(
            lambda g: same_week_day_hour_lag(g[datetime_colname],
                                             g['DEMAND'],
                                             output_colname='LoadLag'))
    same_week_day_hour_load_lag.reset_index(inplace=True)

    # Temperature lags, can serve as a rough temperature forecast
    same_day_hour_drewpnt_lag = \
        output_df[[datetime_colname, 'DewPnt', 'Zone']].groupby('Zone').apply(
            lambda g: same_day_hour_lag(g[datetime_colname], g['DewPnt'],
                                        output_colname='DewPntLag'))
    same_day_hour_drewpnt_lag.reset_index(inplace=True)

    same_day_hour_drybulb_lag = \
        output_df[[datetime_colname, 'DryBulb', 'Zone']].groupby('Zone').apply(
            lambda g: same_day_hour_lag(g[datetime_colname], g['DryBulb'],
                                        output_colname='DryBulbLag'))
    same_day_hour_drybulb_lag.reset_index(inplace=True)

    output_df = reduce(
        lambda left, right: pd.merge(left, right, on=[datetime_colname, 'Zone']),
        [output_df, same_week_day_hour_load_lag,
         same_day_hour_drewpnt_lag, same_day_hour_drybulb_lag])

    if one_hot_encode:
        one_hot_encode = \
            pd.get_dummies(output_df, columns=categorical_columns)

        output_df = output_df.merge(one_hot_encode)
        output_df.drop(categorical_columns, axis=1, inplace=True)

    return output_df
