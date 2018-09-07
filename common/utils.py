import datetime
import pandas as pd

ALLOWED_TIME_COLUMN_TYPES = [pd.Timestamp, pd.DatetimeIndex,
                             datetime.datetime, datetime.date]


def is_datetime_like(x):
    return any(isinstance(x, col_type)
               for col_type in ALLOWED_TIME_COLUMN_TYPES)

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