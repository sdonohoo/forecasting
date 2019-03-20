"""
This is a base class for all time series featurizers.
"""
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTSFeaturizer(BaseEstimator, TransformerMixin):
    def parse_tsdf_config(self, df_config):
        self.time_col_name = df_config['time_col_name']
        self.target_col_name = df_config['target_col_name']
        if isinstance(df_config['ts_id_col_names'], list):
            self.ts_id_col_names = df_config['ts_id_col_names']
        else:
            self.ts_id_col_names = [df_config['ts_id_col_names']]
        self.frequency = df_config['frequency']
        self.time_format = df_config['time_format']

    def fit(self, X, y=None):
        """To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this transformer"""
        return self
