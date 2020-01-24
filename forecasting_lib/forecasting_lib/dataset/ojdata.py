# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import subprocess
import pandas as pd

import forecasting_lib.dataset.benchmark_settings as bs


DATA_FILE_LIST = ["yx.csv", "storedemo.csv"]
SCRIPT_NAME = "download_oj_data.R"


def download_ojdata(dest_dir):
    """Downloads Orange Juice dataset.

    Args:
        dest_dir (str): Directory path for the downloaded file
    """
    maybe_download(dest_dir=dest_dir)


def maybe_download(dest_dir):
    """Download a file if it is not already downloaded.
    
    Args:
        dest_dir (str): Destination directory
        
    Returns:
        str: File path of the file downloaded.
    """
    # Check if data files exist
    data_exists = True
    for f in DATA_FILE_LIST:
        file_path = os.path.join(dest_dir, f)
        data_exists = data_exists and os.path.exists(file_path)

    if not data_exists:
        # Call data download script
        print("Starting data download ...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCRIPT_NAME)
        try:
            subprocess.call(["Rscript", script_path, dest_dir])
        except subprocess.CalledProcessError as e:
            print(e.output)
    else:
        print("Data already exists at the specified location.")


def split_train_test(data_dir, write_csv=False):
    """Generate training, testing, and auxiliary datasets. Training data includes the historical 
    sales and external features; testing data contains the future sales and external features; 
    auxiliary data includes the future price, deal, and advertisement information which can be 
    used for making predictions (we assume such auxiliary information is available at the time 
    when we generate the forecasts).

    You can use this script in either of the following two ways
    1. Import the serve_folds module from this script to generate the training and testing data for
    each forecast period on the fly
    2. Run the script using the syntax below
       python serve_folds [-h] [--test] [--save]
    where if '--test' is specified a quick test of serve_folds module will run and furthermore if
    `--save' is specified the training and testing data will be saved as csv files. Note that '--save'
    is effective only if '--test' is specified. This means that you need to run
       python serve_folds --test --save
    to get the output data files stored in /train and /test folders under the data directory.
    Note that train_*.csv files in /train folder contain all the features in the training period
    and aux_*.csv files in /train folder contain all the features except 'logmove', 'constant',
    'profit' up until the forecast period end week. Both train_*.csv and aux_*csv can be used for
    generating forecasts in each round. However, test_*.csv files in /test folder can only be used
    for model performance evaluation.

    Example:
        if __name__ == "__main__":
            # Test serve_folds
            parser = argparse.ArgumentParser()
            parser.add_argument("--test", help="Run the test of serve_folds function", action="store_true")
            parser.add_argument("--save", help="Write training and testing data into csv files", action="store_true")
            parser.add_argument("--data-dir", help="The location of the downloaded oj data")
            args = parser.parse_args()

            if args.test:
                for train, test, aux in split_train_test(data_dir=args.data_dir, write_csv=args.save):
                    print("Training data size: {}".format(train.shape))
                    print("Testing data size: {}".format(test.shape))
                    print("Auxiliary data size: {}".format(aux.shape))
                    print("Minimum training week number: {}".format(min(train["week"])))
                    print("Maximum training week number: {}".format(max(train["week"])))
                    print("Minimum testing week number: {}".format(min(test["week"])))
                    print("Maximum testing week number: {}".format(max(test["week"])))
                    print("Minimum auxiliary week number: {}".format(min(aux["week"])))
                    print("Maximum auxiliary week number: {}".format(max(aux["week"])))
                    print("")

    Args:
        write_csv (Boolean): Whether to write the data files or not
    """
    # Read sales data into dataframe
    sales = pd.read_csv(os.path.join(data_dir, "yx.csv"), index_col=0)

    if write_csv:
        TRAIN_DATA_DIR = os.path.join(data_dir, "train")
        TEST_DATA_DIR = os.path.join(data_dir, "test")
        if not os.path.isdir(TRAIN_DATA_DIR):
            os.mkdir(TRAIN_DATA_DIR)
        if not os.path.isdir(TEST_DATA_DIR):
            os.mkdir(TEST_DATA_DIR)

    for i in range(bs.NUM_ROUNDS):
        data_mask = (sales.week >= bs.TRAIN_START_WEEK) & (sales.week <= bs.TRAIN_END_WEEK_LIST[i])
        train = sales[data_mask].copy()
        data_mask = (sales.week >= bs.TEST_START_WEEK_LIST[i]) & (sales.week <= bs.TEST_END_WEEK_LIST[i])
        test = sales[data_mask].copy()
        data_mask = (sales.week >= bs.TRAIN_START_WEEK) & (sales.week <= bs.TEST_END_WEEK_LIST[i])
        aux = sales[data_mask].copy()
        aux.drop(["logmove", "constant", "profit"], axis=1, inplace=True)
        if write_csv:
            train.to_csv(os.path.join(TRAIN_DATA_DIR, "train_round_" + str(i + 1) + ".csv"))
            test.to_csv(os.path.join(TEST_DATA_DIR, "test_round_" + str(i + 1) + ".csv"))
            aux.to_csv(os.path.join(TRAIN_DATA_DIR, "aux_round_" + str(i + 1) + ".csv"))
        yield train, test, aux
