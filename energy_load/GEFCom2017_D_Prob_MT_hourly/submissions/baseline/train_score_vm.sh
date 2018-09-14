#!/bin/bash
path=energy_load/GEFCom2017_D_Prob_MT_hourly
for i in `seq 1 5`;
do
    echo "Run $i"
    start=`date +%s`
    echo 'Creating features...'
    python $path/submissions/baseline/feature_engineering.py --submission baseline

    echo 'Training and predicting...'
    Rscript $path/submissions/baseline/train_predict.R $i

    # echo 'Evaluating model quality...'
    # python $path/common/evaluate.py submissions/baseline/submission_seed_$i.csv
    end=`date +%s`
    echo 'Running time '$((end-start))' seconds'
done
