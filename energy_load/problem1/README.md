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
   an [Azure Linux Deep Learning Virtual Machine (DLVM)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu) with Standard NC12 instance. The DLVM will have Docker and [NVIDIA Container Runtime for Docker](https://github.com/NVIDIA/nvidia-docker) automatically installed. If you need to install them manually, you can follow the steps for configuring the machine provided by MLPerf [here](https://github.com/mlperf/reference/tree/master/image_classification#steps-to-configure-machine) (Please follow the steps until the last two for cloning the Github repo of MLPerf). 

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
    You need to have Pandas package installed to run the second command. TODO: wrap the above into `download_data.sh` file

4. Log into Azure Container Registry  
   We use Azure Container Registry (ACR) to store and manage Docker images. You can log into the ACR by
   ```bash
   docker login --username tsperf --password <ACR Access Key> tsperf.azurecr.io
   ```

5. Pull a Docker image from ACR   
   You can pull a Docker image associated with a certain benchmark by
   ```bash
   docker pull tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1
   ```

6. Create a Docker container   
   After pulling the image, you can create a Docker container that runs the image by 
   ```bash
   docker run -it -v ~/TSPerf:/TSPerf --name submission1_container tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1
   ```
   Note that you need to mount `/TSPerf` folder (the one you cloned) to the container so that you will have access to the source code in the container.
   

7. Run benchmarks inside the Docker container   
   In the container, you can go to the `/TSPerf` folder and run the script for model training and scoring by
   ```bash
   python3 ./energy_load/problem1/benchmarks/submission1/train_score.py
   ```
   This will generate a `submission.csv` file under the `/benchmarks/submission1` folder. Then, you can evaluate the forecasting results by 
   ```bash
   python3 energy_load/problem1/common/evaluate.py 'energy_load/problem1/benchmarks/submission1/submission.csv'
   ```
   The above command will output the evaluation metric of this submission. 


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

6. Include a Dockerfile containing all dependencies for running your benchmark (see [./benchmarks/submission1/Dockerfile](./benchmarks/submission1/Dockerfile) for an example). The Dockerfile can point to a `.txt` file which contains a list of necessary packages. 

7. Create a Docker image and push it to the ACR   
   To create your Docker image, you can go to `/benchmarks/submission1/` folder and run the following command   
   ```bash
   docker build -t submission1_image .
   ```
   Then, you can push the image to ACR by executing
   ```bash
   docker tag submission1_image tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1
   docker push tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1
   ```
   Note that you will need to log into the ACR before pushing the image.

8. Create a README file in the submission folder documenting the reported model performance and your approach (see [./benchmarks/submission1/README.md](./benchmarks/submission1/README.md) for an example)

9. Create pull request for review

10. ...


# 4. Reproducibility

To ensure the reproducibility of the benchmark results, you need to    

0. Submit all source code for generating the forecasting results
1. Specify the random seeds according to instructions
2. Report results afer repeated runs of the benchmark
3. Include a Dockerfile containing all dependencies
4. Provide instructions about hyperparameter tuning (optional)

# 5. Quality

**Evaluation metrics**: sMAPE

**Minimum performance**: 12.37%

...

# 6. Leaderboard

| **Submission** | **Model description** | **sMAPE** |
| -------------- | --------------------- | --------- |
| [submission1](./benchmarks/submission1) | seasonal average for month, day of week and hour | 0.1237 |