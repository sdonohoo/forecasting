"""
This file is shared by all submissions to get benchmark paths and add the
TSPerf root path to sys.path.
"""
import os, sys

SUBMISSIONS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SUBMISSIONS_DIR)
TSPERF_DIR = os.path.dirname(os.path.dirname(BENCHMARK_DIR))
BENCHMARK_DATA_DIR = os.path.join(BENCHMARK_DIR, 'data')

if TSPERF_DIR not in sys.path:
    sys.path.insert(0, TSPERF_DIR)
