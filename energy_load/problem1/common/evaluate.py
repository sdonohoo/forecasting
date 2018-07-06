import os
import sys
import pandas as pd
sys.path.append('.')
from common.metrics import sMAPE


def evaluate(submission_file):
    
    # Read submission file and actuals data. Compute and report
    # symmetric MAPE
    submission = pd.read_csv(submission_file, parse_dates=['timestamp'])
    actuals_file = os.path.join('data', 'energy_load', 'energy_load.csv')
    actuals = pd.read_csv(actuals_file, index_col=0, parse_dates=True)

    evaluation = actuals[actuals.task>1][['LOAD']].copy()
    evaluation['timestamp'] = evaluation.index

    evaluation = pd.merge(evaluation, submission, how='left')

    print(sMAPE(evaluation['prediction'], evaluation['LOAD']))


if __name__=="__main__":
    
    submission_file = sys.argv[1]
    evaluate(submission_file)
