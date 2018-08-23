"""
This file is shared by all submissions to get benchmark paths and add the
benchmark common folder to sys.path.
"""
import os, sys

SUBMISSIONS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SUBMISSIONS_DIR)
BENCHMARK_COMMON_DIR = os.path.join(BENCHMARK_DIR, 'common')
BENCHMARK_DATA_DIR = os.path.join(BENCHMARK_DIR, 'data')

if BENCHMARK_COMMON_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_COMMON_DIR)
