import datetime
import pandas as pd

ALLOWED_TIME_COLUMN_TYPES = [pd.Timestamp, pd.DatetimeIndex,
                             datetime.datetime, datetime.date]

def is_datetime_like(x):
    return any(isinstance(x, col_type)
               for col_type in ALLOWED_TIME_COLUMN_TYPES)
