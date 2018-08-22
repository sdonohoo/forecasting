"""
This script adds "/TSPerf/energy_load/GEFCom2017-D_Prob_MT_hourly/submissions"
to the system path. So that benchmark_paths.py can be imported by submission
scripts.
"""
import os, sys
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_SUBMISSIONS_DIR = os.path.dirname(_CUR_DIR)

if _SUBMISSIONS_DIR not in sys.path:
    sys.path.insert(0, _SUBMISSIONS_DIR)
