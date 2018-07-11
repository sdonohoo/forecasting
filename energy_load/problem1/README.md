# 1. Problem

Electricity load point forecasting on GEFCom2014 competition data. Generate forecasts of hourly load for one month ahead.

|  |  |
| ----------------------------------- | - |
| **Number of time series**           | 1 |
| **Forecast generation frequency**   | monthly |
| **Forecast data frequency**         | hourly |
| **Forecast type**                   | point |
| **Available model features**        | *LOAD*: historical hourly electricity load values<br/>*w1-25*: historical hourly temperature values from 25 weather stations in the region. No forecasted temperature values are available at prediction time |

TODO: add public holiday features to dataset (this is allowed in GEFCom2014 competition so we should include it)

### Dataset attribution:

    Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli and Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

# 2. Instructions to run benchmarks

1. Steps to configure machine  
   You can start with an Ubuntu 16.04 machine (12 CPUs, 2 x K80 GPUs, 112GB RAM, 680GB disk) or  
   an [Azure Linux Deep Learning Virtual Machine (DLVM)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu) with Standard NC12 instance. The DLVM will have Docker and NVIDIA driver automatically installed.  

2. Clone the Github repo to your machine by  
   ```bash
   git clone https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_git/TSPerf
   ```

3. Download and preprocess the data using the following commands

    ```bash
    # from the TSPerf root directory
    cd TSPerf
    python energy_load/problem1/common/get_data.py
    ```
    TODO: wrap the above into `download_data.sh` file

4. Log into Azure Container Registry  
   We use Azure Container Registry (ACR) to store and manage Docker images. You can log into the ACR by
   ```bash
   docker login --username tsperf --password <ACR Access Key> tsperf.azurecr.io
   ```

5. Pull a Docker image from ACR

6. Create a Docker container with `/TSPerf` mounted to the container

7. Run benchmarks inside the Docker container



# 3. Instructions to submit new benchmarks

1. Create a new git branch
2. Create a new numbered submission folder under ./energy_load/benchmarks. Pick the next number available
3. Download and preprocess the problem dataset as above

    For this problem, you are provided successive folds of training data. The goal is to generate forecasts for the following periods, using the available training data:

    | **Fold** | **Train period start** | **Train period end** | **Forecast period start** | **Forecast period end** |
    | -------- | --------------- | ------------------ | ------------------------- | ----------------------- |
    | 1 | 2001-01-01 01:00:00 | 2010-10-01 00:00:00 | 2010-10-01 01:00:00 | 2010-11-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2010-11-01 00:00:00 | 2010-10-01 01:00:00 | 2010-12-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2010-12-01 00:00:00 | 2010-10-01 01:00:00 | 2011-01-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2011-01-01 00:00:00 | 2010-10-01 01:00:00 | 2011-02-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2011-02-01 00:00:00 | 2010-10-01 01:00:00 | 2011-03-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2011-03-01 00:00:00 | 2010-10-01 01:00:00 | 2011-04-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2011-04-01 00:00:00 | 2010-10-01 01:00:00 | 2011-05-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2011-05-01 00:00:00 | 2010-10-01 01:00:00 | 2011-06-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2011-06-01 00:00:00 | 2010-10-01 01:00:00 | 2011-07-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2011-07-01 00:00:00 | 2010-10-01 01:00:00 | 2011-08-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2011-08-01 00:00:00 | 2010-10-01 01:00:00 | 2011-09-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2011-09-01 00:00:00 | 2010-10-01 01:00:00 | 2011-10-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2011-10-01 00:00:00 | 2010-10-01 01:00:00 | 2011-11-01 00:00:00 |
    | 1 | 2001-01-01 01:00:00 | 2011-11-01 00:00:00 | 2010-10-01 01:00:00 | 2011-12-01 00:00:00 |

4. To submit your solution you must create a script (in any language) that includes all code necessary to train your model and produce predictions for all forecasted periods in the format:

    | timestamp | prediction |
    | --------- | ---------- |
    | ... | ... |

    The python helper function common/serve_folds.py can be used to serve up successive training and test period folds (see [./benchmarks/submission1/train_score.py](./benchmarks/submission1/train_score.py) for an example of how to use this). You are not obliged to use this function in your submission. You do not have to retrain or retune your model after each successive fold if it does not benefit your model performance. Forecasts **must not** be generated from models that have been trained on data of the forecast period or later. Submission code will be inspected to enforce this.

5. Once you have generated your submission file, you can evaluate the model's performance with
    ```
    python energy_load/problem1/common/evaluate.py \ 
    energy_load/problem1/benchmarks/<submission_dir>/submission.csv
    ```
6. Create a README file in the submission folder documenting the reported model performance and your approach (see [./benchmarks/submission1/README.md](./benchmarks/submission1/README.md) for an example)
7. Include a dockerfile containing the image which includes all dependencies for running your benchmark (TODO: create example)
7. Create pull request for review
8. ...

# 4. Quality

**Evaluation metrics**: sMAPE

**Minimum performance**: 12.37%

...

# 5. Leaderboard

| **Submission** | **Model description** | **sMAPE** |
| -------------- | --------------------- | --------- |
| [submission1](./benchmarks/submission1) | seasonal average for month, day of week and hour | 0.1237 |