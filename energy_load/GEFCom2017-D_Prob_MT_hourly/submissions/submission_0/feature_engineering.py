import os
import pandas as pd

import localpath
from benchmark_paths import BENCHMARK_DATA_PATH
from create_features import create_features
from utils import split_train_test, is_datetime_like

DATETIME_COLNAME = 'Datetime'
HOLIDAY_COLNAME = 'Holiday'
FULL_DATA_FILE = 'full_data.csv'
FULL_DATA_PATH = os.path.join(BENCHMARK_DATA_PATH, FULL_DATA_FILE)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'data/features')

print(OUTPUT_DIR)

def main(input_file, output_dir, datetime_colname, holiday_colname):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    df = pd.read_csv(input_file)

    df_features = create_features(df, datetime_colname=datetime_colname,
                                  holiday_colname=holiday_colname)
    if not is_datetime_like(df_features[DATETIME_COLNAME]):
        df_features[DATETIME_COLNAME] = \
            pd.to_datetime(df_features[DATETIME_COLNAME])
    df_features.set_index(DATETIME_COLNAME, inplace=True)
    split_train_test(df_features, output_dir)


if __name__ == '__main__':
    main(input_file=FULL_DATA_PATH,
         output_dir=OUTPUT_DIR,
         datetime_colname=DATETIME_COLNAME,
         holiday_colname=HOLIDAY_COLNAME)