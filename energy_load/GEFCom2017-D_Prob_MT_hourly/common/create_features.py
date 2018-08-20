"""
This file contains helper functions for creating features on the GEFCom2017
dataset.
"""

from datetime import timedelta
import pandas as pd

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
            datetime_col =  pd.to_datetime(df[datetime_colname])
        except:
            raise Exception('Column or index {0} can not be converted to '
                            'datetime type.'.format(datetime_colname))
    return datetime_col

def day_type(datetime_col, holiday_col=None):

    datetype = pd.DataFrame({'DateType':datetime_col.dt.dayofweek})
    datetype.replace({'DateType': WEEK_DAY_TYPE_MAP}, inplace=True)

    if holiday_col:
        holiday_mask = holiday_col > 0
        datetype[holiday_mask, 'DateType']= HOLIDAY_CODE

        #Create a temporary _Date column to calculate dates near the holidays
        datetype['Date'] = datetime_col.dt.date
        holiday_dates = set(datetype.loc[holiday_mask, 'Date'])

        semi_holiday_dates = [d + SEMI_HOLIDAY_OFFSET for d in holiday_dates] \
                             + [d - SEMI_HOLIDAY_OFFSET for d in holiday_dates]
        semi_holiday_dates = set(semi_holiday_dates)
        semi_holiday_dates = semi_holiday_dates.difference(holiday_dates)

        datetype.loc[datetype['Date'].isin(semi_holiday_dates), 'DateType'] \
            = SEMI_HOLIDAY_CODE

    return datetype['DateType'].values

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
