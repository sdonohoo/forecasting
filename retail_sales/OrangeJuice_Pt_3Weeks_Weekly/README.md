# Problem

Sales forecasting is a key task for the management of retail stores. With the projection of future sales, store managers will be able to optimize 
the inventory based on their business goals. This will generate more profitable order fulfillment and reduce the inventory cost. 

The task of this benchmark is to forecast orange juice sales of different brands for multiple stores with the Orange Juice (OJ) dataset from R package 
`bayesm`. The forecast type is point forecasting. The forecast horizon is 3 weeks ahead and granularity is weekly. There are 12 forecasting rounds, each of 
which involves forecasting the sales during a target period. The training and test data in each round are specified in the subsection [Training and test data 
separation](#training-and-test-data-separation). The table below summarizes the characteristics of this benchmark

|  |  |
| ----------------------------------- | - |
| **Number of time series**           | 913 |
| **Forecast frequency**   | every two weeks |
| **Forecast granularity**         | weekly |
| **Forecast type**                   | point |

# Data

## Dataset attribution

The OJ dataset is from R package [`bayesm`](https://cran.r-project.org/web/packages/bayesm/index.html). 

## Dataset description

This dataset contains the following two tables:

1. Weekly sales of refrigerated orange juice at 83 stores. This table has 106139 rows and 19 columns. It includes weekly sales and prices of 11 orange juice brands as well as information about profit, deal, and advertisement for each brand. 

2. Demographic information on those stores. This table has 83 rows and 13 columns. For every store, the table describes demographic information of its consumers, distance to the nearest warehouse store, average distance to the nearest 5 supermarkets, ratio of its sales to the nearest warehouse store, and ratio of its sales to the average of the nearest 5 stores.

Please see pages 40 and 41 of the [bayesm reference manual](https://cran.r-project.org/web/packages/bayesm/bayesm.pdf) for more details about each data column. 
 

## Training and test data separation

For this benchmark, you are provided successive folds of training data in 12 forecasting rounds. The goal is to generate forecasts for the forecast periods listed in the table below, using the available training data:

| **Round** | **Train period start week** | **Train period end week** | **Forecast period start week** | **Forecast period end week** |
| -------- | --------------- | ------------------ | ------------------------- | ----------------------- |
| 1 | 40 | 95 | 97 | 98 |
| 2 | 40 | 97 | 99 | 100 |
| 3 | 40 | 99 | 101 | 102 |
| 4 | 40 | 101 | 103 | 104 |
| 5 | 40 | 103 | 105 | 106 |
| 6 | 40 | 105 | 107 | 108 |
| 7 | 40 | 107 | 109 | 110 |
| 8 | 40 | 109 | 111 | 112 |
| 9 | 40 | 111 | 113 | 114 |
| 10 | 40 | 113 | 115 | 116 |
| 11 | 40 | 115 | 117 | 118 |
| 12 | 40 | 117 | 119 | 120 |



# Format of Forecasts

The forecasts should be in the following format

| round | store | brand | week | prediction | 
| --------- | ---------- | ---------- | ---------- | ---------- |
| ... | ... | ... | ... | ... |


# Quality

**Evaluation metric**: Mean Absolute Percentage Error (MAPE)  
**Minimum performance**: TBD  

# Instructions to run reference implementation
**TODO: Fill this section when we have a draft reference implementation**

# Instructions to submit new benchmark implementations
**TODO: Fill this section when we have a draft reference implementation**

