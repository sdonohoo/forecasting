"""
This file is shared by all the scripts in the common folder.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
TSPERF_DIR = os.path.dirname(os.path.dirname(BENCHMARK_DIR))
DATA_DIR = os.path.join(BENCHMARK_DIR, 'data')
HOLIDAY_DATA_PATH = os.path.join(TSPERF_DIR, 'common', 'us_holidays.csv')

if TSPERF_DIR not in sys.path:
    sys.path.insert(0, TSPERF_DIR)
