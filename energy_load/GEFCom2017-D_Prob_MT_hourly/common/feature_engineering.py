import os
import pandas as pd

from feature_utils import create_features
from utils import split_train_test, is_datetime_like
from benchmark_paths import DATA_DIR

OUTPUT_DIR = os.path.join(DATA_DIR, 'features')
FULL_DATA_FILE = 'full_data.csv'
FULL_DATA_PATH = os.path.join(DATA_DIR, FULL_DATA_FILE)

DATETIME_COLNAME = 'Datetime'
HOLIDAY_COLNAME = 'Holiday'

def main(input_file, output_dir, datetime_colname, holiday_colname):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    df = pd.read_csv(input_file)

    df_features = create_features(df, datetime_colname=datetime_colname,
                                  holiday_colname=holiday_colname)
    if not is_datetime_like(df_features[datetime_colname]):
        df_features[datetime_colname] = \
            pd.to_datetime(df_features[datetime_colname])
    df_features.set_index(datetime_colname, inplace=True)
    split_train_test(df_features, output_dir)


if __name__ == '__main__':
    main(input_file=FULL_DATA_PATH,
         output_dir=OUTPUT_DIR,
         datetime_colname=DATETIME_COLNAME,
         holiday_colname=HOLIDAY_COLNAME)