"""
This script adds "/TSPerf/energy_load/GEFCom2017-D_Prob_MT_hourly/submissions"
to the system path, so that benchmark_paths.py can be imported by submission
scripts.
"""

import os, sys
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_SUBMISSIONS_DIR = os.path.dirname(_CUR_DIR)
_BENCHMARK_DIR = os.path.dirname(_SUBMISSIONS_DIR)
TSPERF_DIR = os.path.dirname(os.path.dirname(_BENCHMARK_DIR))

if TSPERF_DIR not in sys.path:
    sys.path.insert(0, TSPERF_DIR)
