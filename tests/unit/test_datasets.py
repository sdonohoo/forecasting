import os
import sys
import pytest
import subprocess
import pandas as pd

def test_download_retail_data():
    BENCHMARK_DIR = os.path.join(".", "retail_sales", "OrangeJuice_Pt_3Weeks_Weekly")
    DATA_DIR = os.path.join(BENCHMARK_DIR, "data")
    SCRIPT_PATH = os.path.join(BENCHMARK_DIR, "common", "download_data.r")
    # Call data download script
    subprocess.call(["Rscript", SCRIPT_PATH])
    # Check downloaded data
    sales = pd.read_csv(os.path.join(DATA_DIR, 'yx.csv'), index_col=None)
    assert sales.shape == (106139, 19)
    column_names = ["store", "brand", "week", "logmove", "constant", \
                    "price1", "price2", "price3", "price4", "price5", \
                    "price6", "price7", "price8", "price9", "price10", \
                    "price11", "deal", "feat", "profit"]
    assert list(sales) == column_names
    storedemo = pd.read_csv(os.path.join(DATA_DIR, 'storedemo.csv'), index_col=None)
    assert storedemo.shape == (83, 12)
    column_names = ["STORE", "AGE60", "EDUC", "ETHNIC", "INCOME", \
                    "HHLARGE", "WORKWOM","HVAL150","SSTRDIST", \
                    "SSTRVOL", "CPDIST5", "CPWVOL5"]
    assert list(storedemo) == column_names

def test_download_energy_data():
    BENCHMARK_DIR = os.path.join(".", "energy_load", "GEFCom2017_D_Prob_MT_hourly")
    DATA_DIR = os.path.join(BENCHMARK_DIR, "data")
    SCRIPT_PATH = os.path.join(BENCHMARK_DIR, "common", "download_data.py")
    # Call data download script
    try:
        subprocess.check_call(["python", SCRIPT_PATH], shell=True)
    except subprocess.CalledProcessError as e:
        print("Unable to download valid data from at least one URL.")

