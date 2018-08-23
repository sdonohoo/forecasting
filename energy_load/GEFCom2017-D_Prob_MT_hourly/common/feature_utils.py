"""
This file contains helper functions for creating features on the GEFCom2017
dataset.
"""

from datetime import timedelta
import calendar
import math
import pandas as pd
import numpy as np

from utils import is_datetime_like

# 0: Monday, 2: T/W/TR, 4: F, 5:SA, 6: S
WEEK_DAY_TYPE_MAP = {1:2, 3:2}
HOLIDAY_CODE = 7
SEMI_HOLIDAY_CODE = 8
SEMI_HOLIDAY_OFFSET = timedelta(days=1)


def get_datetime_col(df, datetime_colname):
    if datetime_colname in df.index.names:
        datetime_col = df.index.get_level_values(datetime_colname)
    elif datetime_colname in df.columns:
        datetime_col = df[datetime_colname]
    else:
        raise Exception('Column or index {0} does not exist in the data '
                        'frame'.format(datetime_colname))

    if not is_datetime_like(datetime_col):
        try:
            datetime_col = pd.to_datetime(df[datetime_colname])
        except:
            raise Exception('Column or index {0} can not be converted to '
                            'datetime type.'.format(datetime_colname))
    return datetime_col


def day_type(datetime_col, holiday_col=None):

    datetype = pd.DataFrame({'DayType': datetime_col.dt.dayofweek})
    datetype.replace({'DayType': WEEK_DAY_TYPE_MAP}, inplace=True)

    if holiday_col is not None:
        holiday_mask = holiday_col > 0
        datetype.loc[holiday_mask, 'DayType'] = HOLIDAY_CODE

        #Create a temporary _Date column to calculate dates near the holidays
        datetype['Date'] = datetime_col.dt.date
        holiday_dates = set(datetype.loc[holiday_mask, 'Date'])

        semi_holiday_dates = [d + SEMI_HOLIDAY_OFFSET for d in holiday_dates] \
                             + [d - SEMI_HOLIDAY_OFFSET for d in holiday_dates]
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
                                 'HourOfDay': datetime_col.dt.hour})
    time_of_year['TimeOfYear'] = time_of_year.apply(
        lambda row: (row.DayOfYear - 1) * 24 + row.HourOfDay, axis=1)

    #TODO: Update this based on if the year is leap year
    min_toy = min(time_of_year['TimeOfYear'])
    max_toy = max(time_of_year['TimeOfYear'])

    time_of_year['TimeOfYear'] = \
        (time_of_year['TimeOfYear'] - min_toy)/(max_toy - min_toy)

    return time_of_year['TimeOfYear'].values


def week_of_year(datetime_col):
    return datetime_col.dt.week


def current_year(datetime_col, min_year, max_year):

    year = datetime_col.dt.year
    current_year = (year - min_year)/(max_year - min_year)

    return current_year


def fourier_approximation(t, n, period):
    x = n * 2 * math.pi * t/period
    x_sin = x.apply(lambda i: math.sin(i))
    x_cos = x.apply(lambda i: math.cos(i))

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
                           week_window=1, agg_func='mean'):

    min_time_stamp = min(datetime_col)

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    week_lag_base = 52
    week_lag_last_year = list(range(week_lag_base-week_window,
                              week_lag_base+week_window+1))
    week_lag_all = []
    for y in range(n_years):
        week_lag_all += [x + y * 52 for x in week_lag_last_year]

    week_lag_cols = []
    for w in week_lag_all:
        col_name = 'week_lag_' + str(w)
        week_lag_cols.append(col_name)

        lag_datetime = df.index.get_level_values(0) - timedelta(weeks=w)
        valid_lag_mask = lag_datetime >= min_time_stamp

        df[col_name] = np.nan

        df.loc[valid_lag_mask, col_name] = \
            df.loc[lag_datetime[valid_lag_mask], 'value']

    # Additional aggregation options will be added as needed
    if agg_func == 'mean':
        df['aggregated_lag'] = df[week_lag_cols].mean(axis=1)

    return df['aggregated_lag'].values


def create_features(input_df, datetime_colname,
                    holiday_colname=None, one_hot_encode=True):

    categorical_columns = ['DayType', 'Hour', 'WeekOfYear']

    output_df = input_df.copy()
    datetime_col = get_datetime_col(output_df, datetime_colname)

    output_df['DayType'] = day_type(datetime_col, output_df[holiday_colname])
    output_df['Hour'] = hour_of_day(datetime_col)
    output_df['TimeOfYear'] = time_of_year(datetime_col)
    output_df['WeekOfYear'] = week_of_year(datetime_col)
    output_df['CurrentYear'] = current_year(datetime_col, 2011, 2017)

    if one_hot_encode:
        one_hot_encode = \
            pd.get_dummies(output_df, columns=categorical_columns)

        output_df = output_df.merge(one_hot_encode)
        output_df.drop(categorical_columns, axis=1, inplace=True)

    return output_df
