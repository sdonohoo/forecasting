#!/bin/bash

source activate tsperf

for i in `seq 1 5`;
do 
    echo "Run $i"

    echo 'Creating features...'
    python feature_engineering.py --submission submission_0

    echo 'Training and predicting...'
    Rscript train_predict.R

    echo 'Evaluating model quality...'
    python ../../common/evaluate.py submissions/submission_0/submission.csv
done