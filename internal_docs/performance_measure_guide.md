## Guidance for Performance Measuring

Each benchmark result is the median of five run results produced using the integer random number generator seeds 1 through 5. All five run results must also be reported. The following measurements should be reported:
  * quality of the model
  * running time
  * cloud cost 

### Quality of the Model

Each run must reach a target quality level on the reference implementation quality measure. The time to measure quality is included in the wallclock time.

Please use common utility script `evaluate.py` to get the benchmark quality value (e.g. MAPE) in each run
```bash
python <benchmark directory>/common/evaluate.py <submission directory>/submission.xls
``` 

### Running Time

The wallclock running time of each run should be measured by 
```bash
time -p python <submission directory>/train_score.py
```

### Cloud Cost

Include the total cost of obtaining the median run result using fixed prices for the general public at the time the result is collected. Do not use spot pricing.