# Problem

Probabilistic load forecasting (PLF) has become increasingly important in
power systems planning and operations in recent years. The applications of PLF
include energy production planning, reliability analysis, probabilistic price
forecasting, etc.
The task of this benchmark is to generate probabilistic forecasting of
electricity load on the GEFCom2017 competition qualifying match data. The
forecast horizon is 1~2 months ahead and granularity is hourly, see [Training
and test separation](#training-and-test-data-separation) for details. The
forecasts should be in the form of 9 quantiles, i.e. the 10th, 20th, ... 90th
percentiles, following the format of the provided template file.  There are 10
time series (zones) to forecast, including the 8 ISO New England zones, the
Massachusetts (sum of three zones under Massachusetts), and the total (sum of
the first 8 zones).
The table below summarizes the benchmark problem definition:

|||
| ----------------------------------- | ---- |  
| **Number of time series**           | 10 |
| **Forecast frequency**   | twice every month, mid and end of month |
| **Forecast granularity**         | hourly |
| **Forecast type**                   | probabilistic, 9 quantiles: 10th, 20th, ...90th percentiles|

A template of the submission file can be found [here](https://github.com/Microsoft/Forecasting/blob/master/energy_load/GEFCom2017_D_Prob_MT_hourly/reference/submission.csv)

# Data  
### Dataset attribution
[ISO New England](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info)

### Dataset description

1. The data files can be downloaded from ISO New England website via the
[zonal information page of the energy, load and demand reports](https://www
.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info). If you
are outside United States, you may need a VPN to access the data. Use columns
A, B, D, M and N in the worksheets of "YYYY SMD Hourly Data" files, where YYYY
represents the year. Detailed information of each column can be found in the
"Notes" sheet of the data files.  
The script energy_load/GEFCom2017_D_Prob_MT_hourly/common/download_data.py downloads the load data to energy_load/GEFCom2017_D_Prob_MT_hourly/data/.

2. US Federal Holidays as published via [US Office of Personnel Management](https://www.opm.gov/policy-data-oversight/snow-dismissal-procedures/federal-holidays/).  
This data can be found [here](https://github.com/Microsoft/Forecasting/blob/master/common/us_holidays.csv).

### Data preprocessing (TBD)
The script energy_load/GEFCom2017_D_Prob_MT_hourly/common/extract_data.py
parses the excel files and creates training and testing csv load files. The
following preprocessing steps are performed by this script:  
* Map the holiday names to integers and join holiday data with load data.  
* When the --preprocess argument is True, zero load values are filled by
the values of the same hour of the previous day, outliers caused by end of
Daylight Saving Time are divided by 2.
* In addition to the eight zones in the excel files, 'SEMA', 'WCMA', and 'NEMA'
are aggregated to generate the MA_TOTAL zone and all eight zones are aggregated
to generate the TOTAL zone.

### Training and test data separation
For this problem, you are provided successive folds of training data. The goal
is to generate forecasts for the forecast periods listed in the table below,
using the available training data:

| **Round** | **Train period start** | **Train period end** | **Forecast period start** | **Forecast period end** |
| -------- | --------------- | ------------------ | ------------------------- | ----------------------- |
| 1 | 2011-01-01 01:00:00 | 2016-11-30 00:00:00 | 2017-01-01 01:00:00 | 2017-01-31 00:00:00 |
| 2 | 2011-01-01 01:00:00 | 2016-11-30 00:00:00 | 2017-02-01 01:00:00 | 2017-02-28 00:00:00 |
| 3 | 2011-01-01 01:00:00 | 2016-12-31 00:00:00 | 2017-02-01 01:00:00 | 2017-02-28 00:00:00 |
| 4 | 2011-01-01 01:00:00 | 2016-12-31 00:00:00 | 2017-03-01 01:00:00 | 2017-03-31 00:00:00 |
| 5 | 2011-01-01 01:00:00 | 2017-01-31 00:00:00 | 2017-03-01 01:00:00 | 2017-03-31 00:00:00 |
| 6 | 2011-01-01 01:00:00 | 2017-01-31 00:00:00 | 2017-04-01 01:00:00 | 2017-04-30 00:00:00 |

# Model Evaluation

**Evaluation metric**: Pinball loss  
**Minimum performance**: 81  
