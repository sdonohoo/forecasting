import pandas as pd
from sklearn.base import BaseEstimator


class YearNormalizer(BaseEstimator):
    """
    Temporal feature indicating the position of the year of a record in the
    entire time period under consideration, normalized to be between 0 and 1.

    Args:
        datetime_col: Datetime column.
        min_year: minimum value of year.
        max_year: maximum value of year.

    Returns:
        float: the position of the current year in the min_year:max_year range
    """
    def __init__(self, df_config):
        self.time_col_name = df_config['time_col_name']
        self._is_fit = False

    def fit(self, X, y=None):
        datetime_col = X[self.time_col_name]
        self.min_year = min(datetime_col.dt.year)
        self.max_year = max(datetime_col.dt.year)
        self._is_fit = True

    def transform(self, X):
        datetime_col = X[self.time_col_name]
        year = datetime_col.dt.year

        if self.max_year != self.min_year:
            X['normalized_year'] = \
                (year - self.min_year) / (self.max_year - self.min_year)
        elif self.max_year == self.min_year:
            X['normalized_year'] = 0

        return X


class DateNormalizer(BaseEstimator):
    """
    Temporal feature indicating the position of the date of a record in the
    entire time period under consideration, normalized to be between 0 and 1.

    Args:
        datetime_col: Datetime column.
        min_date: minimum value of date.
        max_date: maximum value of date.
    Returns:
        float: the position of the current date in the min_date:max_date range
    """

    def __init__(self, df_config):
        self.time_col_name = df_config['time_col_name']
        self._is_fit = False

    def fit(self, X, y=None):
        datetime_col = X[self.time_col_name]
        self.min_date = min(datetime_col.dt.date)
        self.max_date = max(datetime_col.dt.date)
        self._is_fit = True

    def transform(self, X):
        datetime_col = X[self.time_col_name]

        date = datetime_col.dt.date
        current_date = (date - self.min_date).apply(lambda x: x.days)

        if self.max_date != self.min_date:
            X['normalized_date'] = \
                current_date / (self.max_date - self.min_date).days
        elif self.max_date == self.min_date:
            X['normalized_date'] = 0

        return X


class DateHourNormalizer(BaseEstimator):
    """
    Temporal feature indicating the position of the hour of a record in the
    entire time period under consideration, normalized to be between 0 and 1.

    Args:
        datetime_col: Datetime column.
        min_datehour: minimum value of datehour.
        max_datehour: maximum value of datehour.
    Returns:
        float: the position of the current datehour in the min_datehour:max_datehour range
    """
    def __init__(self, df_config):
        self.time_col_name = df_config['time_col_name']
        self._is_fit = False

    def fit(self, X, y=None):
        datetime_col = X[self.time_col_name]
        self.min_datehour = min(datetime_col)
        self.max_datehour = max(datetime_col)
        self._is_fit = True

    def transform(self, X):
        datetime_col = X[self.time_col_name]
        current_datehour = (datetime_col - self.min_datehour)\
            .apply(lambda x: x.days * 24 + x.seconds / 3600)

        max_min_diff = self.max_datehour - self.min_datehour

        if max_min_diff != 0:
            X['normalized_datehour'] = \
                current_datehour(
                    max_min_diff.days * 24 + max_min_diff.seconds / 3600)
        elif max_min_diff == 0:
            X['normalized_datehour'] = 0

        return X
