# TSPerf

## Time Series Forecasting Model Benchmarking

A framework for evaluating different approaches to common real world time series forecasting problems.

## Contents

Benchmark approaches are submitted against forecasting problems in the following domains:

- [energy_load](./energy_load) - electricity demand forecasting
- energy_price - electricity price forecasting
- retail_sales - forecasts for sales of products in retail stores
- ...

## Running benchmarks

To run any of the submitted benchmarks in this repository, your system must meet the following requirements:

- Ubuntu 16.04
- ...

and have the following pre-requisites installed:
- git version...
- Anaconda version ...
- docker ...

Clone this repository:
```
git clone ...
```
Run the following commands to install these dependencies:
```
...
```
Submitted benchmarks will have additional dependencies specified in their respective README files.

## Submitting benchmarks

1. Create a new git branch
2. Follow the problem-specific instructions to create a benchmark submission
3. Submit a pull request
4. TSPerf board will review submission
5. PR will be merged if accepted

## Forecasting problems

The table below describes the forecasting problems to be tackled and the best models found to date.

| **Domain** | **Problem** | **Problem description** | **Best model description** |
| ---------- | ----------- | ---------------- | ------------------- | -------------- |
| Energy load | [problem1](./energy_load/problem1/) | 1 month ahead point forecasting | [seasonal average](./energy_load/benchmarks/submission1/) |
| Energy load | problem2 | 1 month ahead probabilistic | ??? |
| Energy load | problem3 | 1 day ahead point | ??? |
| Energy load | problem4 | 1 day ahead probabilistic | ??? |
| Energy price | problem1 | 1 day ahead point | ??? |
| Energy price | problem2 | 1 day ahead probabilistic | ??? |
| Retail sales | problem1 | 1 year ahead point | ??? |
| Retail sales | problem2 | 1 year ahead probabilistic | ??? |

