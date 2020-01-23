# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This module contains benchmark related parameters.

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
TSPERF_DIR = os.path.dirname(os.path.dirname(BENCHMARK_DIR))

SUBMISSIONS_DIR = os.path.join(BENCHMARK_DIR, "submissions")
DATA_DIR = os.path.join(BENCHMARK_DIR, "data")

NUM_ROUNDS = 12
PRED_HORIZON = 3
PRED_STEPS = 2
TRAIN_START_WEEK = 40
TRAIN_END_WEEK_LIST = list(range(135, 159, 2))
TEST_START_WEEK_LIST = list(range(137, 161, 2))
TEST_END_WEEK_LIST = list(range(138, 162, 2))
# The start datetime of the first week in the record
FIRST_WEEK_START = pd.to_datetime("1989-09-14 00:00:00")
