"""
This script evaluates an implementation of the 
OrangeJuice_Pt_3Weeks_Weekly benchmark. It reads in
the test set predictions contained in submission.csv
file that should accompany every reference implementation
and submission. It computes evaluation metrics by
comparing the predictions to the actual true values
contained within the test set.

The script must be executed from the TSPerf root directory.

Arguments:
    submission_file:   relative file path to submission.csv 
        to the reference or submission implementation
"""

import os
import sys
import pandas as pd
sys.path.append('.')
from common.metrics import sMAPE

def read_test_files(benchmark_dir):
    
    test_data_dir = os.path.join(benchmark_dir, "data", "test")
    for rnd in range(1, 13):
        test_file = 'test_round_'+str(rnd)+'.csv'
        test_round = pd.read_csv(os.path.join(test_data_dir, test_file))
        test_round['round'] = rnd
        test_round = test_round[['round', 'store', 'brand', 'week', 'logmove']]
        if rnd > 1:
            test = test.append(test_round)
        else:
            test = test_round.copy()
    return test


def evaluate(submission_file):
    
    benchmark_dir = os.path.join("retail_sales", "OrangeJuice_Pt_3Weeks_Weekly")

    test = read_test_files(benchmark_dir)

    submission = pd.read_csv(submission_file)

    evaluation = pd.merge(submission, test, on=['round', 'store', 'brand', 'week'], how='left')

    print("sMAPE: ", sMAPE(evaluation['prediction'], evaluation['logmove']))



if __name__=="__main__":
    
    submission_file = sys.argv[1]
    evaluate(submission_file)