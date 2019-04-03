# TSPerf

TSPerf is a repository of time-series forecasting models with a comprehensive comparison of their performance over provided benchmark data sets, implemented on Azure. Model implementations are compared by forecasting accuracy, training and scoring time and cost on Azure compute. Each implementation includes all the necessary instructions and tools that ensure its reproducibility. We envision TSPerf to become a central repository of time-series forecasting that provides wide coverage of time-series  algorithms, from the very simple to the state of the art in the industry. The roadmap of TSPerf can be found [here](docs/roadmap.md).


The following table summarizes benchmarks that are currently included in TSPerf.

Benchmark                                   |  Dataset               |  Benchmark directory
--------------------------------------------|------------------------|---------------------------------------------
Probabilistic electricity load forecasting  |  GEFCom2017            |  `energy_load/GEFCom2017-D_Prob_MT_Hourly`
Retail sales forecasting                    |  Orange Juice dataset  |  `retail_sales/OrangeJuice_Pt_3Weeks_Weekly`




A complete documentation of TSPerf, along with the instructions for submitting and reviewing implementations, can be found [here](./docs/tsperf_rules.md). The tables below show performance of implementations that are developed so far. Source code of implementations and instructions for reproducing their performance can be found in submission folders, which are linked in the first column.

## Probabilistic energy forecasting performance board


The following table lists the current submision for the energy forecasting and their respective performances.

Submission Name                                                                 |  Pinball Loss  |  Training and Scoring Time (sec)  |  Training and Scoring Cost($)  |  Architecture                                 |  Framework                         |  Algorithm                            |  Uni/Multivariate  |  External Feature Support
--------------------------------------------------------------------------------|----------------|-----------------------------------|--------------------------------|-----------------------------------------------|------------------------------------|---------------------------------------|--------------------|--------------------------
[Baseline](energy_load%2FGEFCom2017_D_Prob_MT_hourly%2Fsubmissions%2Fbaseline)  |   84.12         |  188                              |  0.0201                       |  Linux DSVM (Standard D8s v3 - Premium SSD)   |  quantreg package of R             |  Linear Quantile Regression           |  Multivariate      |  Yes
[GBM](energy_load%2FGEFCom2017_D_Prob_MT_hourly%2Fsubmissions%2FGBM)            |   78.84         |  269                              |  0.0287                       |  Linux DSVM (Standard D8s v3 - Premium SSD)   |  gbm package of R                  |  Gradient Boosting Decision Tree      |  Multivariate      |  Yes
[QRF](energy_load%2FGEFCom2017_D_Prob_MT_hourly%2Fsubmissions%2Fqrf)            |    76.29         |  20322                            |  17.19                       |   Linux DSVM (F72s v2 - Premium SSD)          |   scikit-garden package of Python  |   Quantile Regression Forest          |   Multivariate     |   Yes
[FNN](energy_load%2FGEFCom2017_D_Prob_MT_hourly%2Fsubmissions%2Ffnn)            |    80.06         |  1085                             |  0.1157                      |   Linux DSVM (Standard D8s v3 - Premium SSD)  |   qrnn package of R                |   Quantile Regression Neural Network  |   Multivariate     |   Yes

The following chart compares the submissions performance on accuracy in Pinball Loss vs. Training and Scoring cost in $:

 ![EnergyPBLvsTime](./docs/images/Energy-Cost.png)




## Retail sales forecasting performance board


The following table lists the current submision for the retail forecasting and their respective performances.


Submission Name                                                                             |  MAPE (%)  |  Training and Scoring Time (sec)  |  Training and Scoring Cost ($)  |  Architecture                                |  Framework                   |  Algorithm                                                          |  Uni/Multivariate  |  External Feature Support
--------------------------------------------------------------------------------------------|------------|-----------------------------------|---------------------------------|----------------------------------------------|------------------------------|---------------------------------------------------------------------|--------------------|--------------------------
[Baseline](retail_sales%2FOrangeJuice_Pt_3Weeks_Weekly%2Fsubmissions%2Fbaseline)            |  109.67    |  114.06                           |  0.003                          |  Linux DSVM(Standard D2s v3 - Premium SSD)   |  forecast package of R       |  Naive Forecast                                                     |  Univariate        |  No
[AutoARIMA](retail_sales%2FOrangeJuice_Pt_3Weeks_Weekly%2Fsubmissions%2FARIMA)              |  70.80     |  265.94                           |  0.0071                         |  Linux DSVM(Standard D2s v3 - Premium SSD)   |  forecast package of R       |  Auto ARIMA                                                         |  Multivariate      |  Yes
[ETS](retail_sales%2FOrangeJuice_Pt_3Weeks_Weekly%2Fsubmissions%2FETS)                      |  70.99     |  277                              |  0.01                           |  Linux DSVM(Standard D2s v3 - Premium SSD)   |  forecast package of R       |  ETS                                                                |  Multivariate      |  No
[MeanForecast](retail_sales%2FOrangeJuice_Pt_3Weeks_Weekly%2Fsubmissions%2FMeanForecast)    |  70.74     |  69.88                            |  0.002                          |  Linux DSVM(Standard D2s v3 - Premium SSD)   |  forecast package of R       |  Mean forecast                                                      |   Univariate       |  No
[SeasonalNaive](retail_sales%2FOrangeJuice_Pt_3Weeks_Weekly%2Fsubmissions%2FSeasonalNaive)  |  165.06    |  160.45                           |  0.004                          |  Linux DSVM(Standard D2s v3 - Premium SSD)   |  forecast package of R       |  Seasonal Naive                                                     |  Univariate        |  No
[LightGBM](retail_sales%2FOrangeJuice_Pt_3Weeks_Weekly%2Fsubmissions%2FLightGBM)            |  36.28     |  625.10                           |  0.0167                         |  Linux DSVM (Standard D2s v3 - Premium SSD)  |  lightGBM package of Python  |  Gradient Boosting Decision Tree                                    |  Multivariate      |  Yes
[DilatedCNN](retail_sales%2FOrangeJuice_Pt_3Weeks_Weekly%2Fsubmissions%2FDilatedCNN)        |  37.09     |  413                              |  0.1032                         |  Ubuntu VM(NC6 - Standard HDD)               |  Keras and Tensorflow        |  Python + Dilated convolutional neural network                      |   Multivariate     |  Yes
[RNN Encoder-Decoder](retail_sales%2FOrangeJuice_Pt_3Weeks_Weekly%2Fsubmissions%2FRNN)      |  37.68     |  669                              |  0.2                            |  Ubuntu VM(NC6 - Standard HDD)               |  Tensorflow                  |  Python + Encoder-decoder architecture of recurrent neural network  |   Multivariate     |  Yes






The following chart compares the submissions performance on accuracy in %MAPE vs. Training and Scoring cost in $:

 
![EnergyPBLvsTime](./docs/images/Retail-Cost.png)

## Build Status


| Build Type | Branch | Status |  | Branch | Status | 
| --- | --- | --- | --- | --- | --- | 
| **Python Linux CPU** |  master | [![Build Status](https://dev.azure.com/best-practices/forecasting/_apis/build/status/python_unit_tests_base?branchName=master)](https://dev.azure.com/best-practices/forecasting/_build/latest?definitionId=12&branchName=master)  | | staging | [![Build Status](https://dev.azure.com/best-practices/forecasting/_apis/build/status/python_unit_tests_base?branchName=chenhui/python_test_pipeline)](https://dev.azure.com/best-practices/forecasting/_build/latest?definitionId=12&branchName=chenhui/python_test_pipeline) | 
| **R Linux CPU** |  master | [![Build Status](https://dev.azure.com/best-practices/forecasting/_apis/build/status/Forecasting/r_unit_tests_prototype?branchName=master)](https://dev.azure.com/best-practices/forecasting/_build/latest?definitionId=9&branchName=master)  | | staging | [![Build Status](https://dev.azure.com/best-practices/forecasting/_apis/build/status/Forecasting/r_unit_tests_prototype?branchName=zhouf/r_test_pipeline)](https://dev.azure.com/best-practices/forecasting/_build/latest?definitionId=9&branchName=zhouf/r_test_pipeline) | 

