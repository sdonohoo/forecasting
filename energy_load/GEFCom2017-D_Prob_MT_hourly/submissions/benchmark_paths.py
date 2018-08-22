import os, sys

SUBMISSIONS_PATH = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_PATH = os.path.dirname(SUBMISSIONS_PATH)
BENCHMARK_COMMON_PATH = os.path.join(BENCHMARK_PATH, 'common')
BENCHMARK_DATA_PATH = os.path.join(BENCHMARK_PATH, 'data')

if BENCHMARK_COMMON_PATH not in sys.path:
    sys.path.insert(0, BENCHMARK_COMMON_PATH)