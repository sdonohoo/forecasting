from sklearn.base import BaseEstimator
import warnings
import pandas as pd
from ..utils import is_datetime_like


class YearNormalizer(BaseEstimator):
    """
    Creates a temporal feature indicating the position of the year of a record
    in the entire time period under consideration, normalized to be between
    0 and 1.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.

    """
    def __init__(self, df_config):
        self.time_col_name = df_config['time_col_name']
        self.time_format = df_config['time_format']
        self._is_fit = False

    def fit(self, X, y=None):
        datetime_col = X[self.time_col_name]
        if not is_datetime_like(datetime_col):
            datetime_col = pd.to_datetime(datetime_col, format=self.time_format)
        self.min_year = datetime_col.dt.year.min()
        self.max_year = datetime_col.dt.year.max()
        self._is_fit = True

        return self

    def transform(self, X):
        if not self._is_fit:
            raise Exception('The featurizer needs to be fitted first to be '
                            'used to transform data.')

        if 'normalized_year' in X.columns:
            warnings.warn('Column {} is already in the data frame, '
                          'it will be overwritten.'.format('normalized_year'))

        datetime_col = X[self.time_col_name]
        if not is_datetime_like(datetime_col):
            datetime_col = pd.to_datetime(datetime_col, format=self.time_format)
        year = datetime_col.dt.year
        if self.max_year != self.min_year:
            X['normalized_year'] = \
                (year - self.min_year) / (self.max_year - self.min_year)
        else:
            X['normalized_year'] = 0

        return X


class DateNormalizer(BaseEstimator):
    """
    Creates a temporal feature indicating the position of the date of a record
    in the entire time period under consideration, normalized to be between
    0 and 1.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
    """

    def __init__(self, df_config):
        self.time_col_name = df_config['time_col_name']
        self.time_format = df_config['time_format']
        self._is_fit = False

    def fit(self, X, y=None):
        datetime_col = X[self.time_col_name]
        if not is_datetime_like(datetime_col):
            datetime_col = pd.to_datetime(datetime_col, format=self.time_format)
        self.min_date = datetime_col.dt.date.min()
        self.max_date = datetime_col.dt.date.max()
        self._is_fit = True

        return self

    def transform(self, X):
        if not self._is_fit:
            raise Exception('The featurizer needs to be fitted first to be '
                            'used to transform data.')

        if 'normalized_date' in X.columns:
            warnings.warn('Column {} is already in the data frame, '
                          'it will be overwritten.'.format('normalized_date'))

        datetime_col = X[self.time_col_name]
        if not is_datetime_like(datetime_col):
            datetime_col = pd.to_datetime(datetime_col, format=self.time_format)
        date = datetime_col.dt.date
        current_date = (date - self.min_date).apply(lambda x: x.days)

        if self.max_date != self.min_date:
            X['normalized_date'] = \
                current_date / (self.max_date - self.min_date).days
        else:
            X['normalized_date'] = 0

        return X


class DateHourNormalizer(BaseEstimator):
    """
    Creates a temporal feature indicating the position of the hour of a record
    in the entire time period under consideration, normalized to be between
    0 and 1.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
    """
    def __init__(self, df_config):
        self.time_col_name = df_config['time_col_name']
        self.time_format = df_config['time_format']
        self._is_fit = False

    def fit(self, X, y=None):
        datetime_col = X[self.time_col_name]
        if not is_datetime_like(datetime_col):
            datetime_col = pd.to_datetime(datetime_col, format=self.time_format)
        self.min_datehour = datetime_col.min()
        self.max_datehour = datetime_col.max()
        self._is_fit = True

        return self

    def transform(self, X):
        if not self._is_fit:
            raise Exception('The featurizer needs to be fitted first to be '
                            'used to transform data.')

        if 'normalized_datehour' in X.columns:
            warnings.warn(
                'Column {} is already in the data frame, '
                'it will be overwritten.'.format('normalized_datehour'))

        datetime_col = X[self.time_col_name]
        if not is_datetime_like(datetime_col):
            datetime_col = pd.to_datetime(datetime_col, format=self.time_format)
        current_datehour = (datetime_col - self.min_datehour)\
            .apply(lambda x: x.days * 24 + x.seconds / 3600)

        max_min_diff = self.max_datehour - self.min_datehour

        if max_min_diff != 0:
            X['normalized_datehour'] = \
                current_datehour/(
                        max_min_diff.days * 24 + max_min_diff.seconds / 3600)
        else:
            X['normalized_datehour'] = 0

        return X
