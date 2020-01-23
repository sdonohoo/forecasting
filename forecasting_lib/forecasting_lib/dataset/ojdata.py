# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import subprocess

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


if __name__ == "__main__":
    data_dir = "ojdata12"
    download_ojdata(data_dir)
