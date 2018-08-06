# Problem

Probabilistic load forecasting (PLF) has become increasingly important in power systems planning and operations in recent years. The applications of PLF include unit commitment planning, reliability analysis, probabilistic price forecasting, etc.  
The task of this benchmark is to generate probabilistic forecasting of electricity load on the GEFCom2017 competition qualifying match data. The
forecast horizon is 1~2 months ahead and granularity is hourly, see [Training and test separation](#training-and-test-data-separation) for details. The forecasts should be in the form of 9 quantiles, i.e. the 10th, 20th, ... 90th percentiles, following the format of the provided template file.  There are 10 time series (zones) to forecast, including the 8 ISO New England zones, the Massachusetts (sum of three zones under Massachusetts), and the total (sum of the first 8 zones).  
The table below summarizes the benchmark problem definition:

|  |  |
| ----------------------------------- | - |
| **Number of time series**           | 10 |
| **Forecast frequency**   | twice every month, mid and end of month |
| **Forecast granularity**         | hourly |
| **Forecast type**                   | probabilistic, 9 quantiles: 10th, 20th, ...90th percentiles|
**TODO: create template of output file**
# Data  
### Dataset attribution
[ISO New England](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info)

### Dataset description

1. The data files can be downloaded from ISO New England website via the [zonal information page of the energy, load and demand reports](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info). If you are outside United States, you may need a VPN to access the data. Use columns A, B, D, M and N in the worksheets of "YYYY SMD Hourly Data" files, where YYYY represents the year. Detailed information of each column can be found in the "Notes" sheet of the data files.

2. US Federal Holidays as published via [US Office of Personnel Management](https://www.opm.gov/policy-data-oversight/snow-dismissal-procedures/federal-holidays/).

### Data preprocessing (TBD)
**TODO: Scripts for reading data, aggregating time series, and extract basic features.**

### Training and test data separation
For this problem, you are provided successive folds of training data. The goal is to generate forecasts for the forecast periods listed in the table below, using the available training data:

| **Round** | **Train period start** | **Train period end** | **Forecast period start** | **Forecast period end** |
| -------- | --------------- | ------------------ | ------------------------- | ----------------------- |
| 1 | 2011-01-01 01:00:00 | 2016-12-15 00:00:00 | 2017-01-01 01:00:00 | 2017-01-31 00:00:00 |
| 2 | 2011-01-01 01:00:00 | 2016-12-31 00:00:00 | 2017-02-01 01:00:00 | 2017-02-28 00:00:00 |
| 3 | 2011-01-01 01:00:00 | 2017-01-15 00:00:00 | 2017-02-01 01:00:00 | 2017-02-28 00:00:00 |
| 4 | 2011-01-01 01:00:00 | 2017-01-31 00:00:00 | 2017-03-01 01:00:00 | 2017-03-31 00:00:00 |
| 5 | 2011-01-01 01:00:00 | 2017-02-14 00:00:00 | 2017-03-01 01:00:00 | 2017-03-31 00:00:00 |
| 6 | 2011-01-01 01:00:00 | 2017-02-28 00:00:00 | 2017-04-01 01:00:00 | 2017-04-30 00:00:00 |

**TODO: We need to provide code to do this separation.**

# Model (TBD)

# Quality

**Evaluation metric**: Pinball loss function  
**Minimum performance**: TBD  
**Evaluation frequency**: TBD  
**Evaluation thoroughness**: TBD  

**TODO: Write script to compute pinball loss function.**

# Instructions to run reference implementation
**TODO: Revise this section when we have a draft reference implementation. We will need to update the script directories and maybe refine environment setup.**

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
    You need to have Pandas package installed to run the second command.  
    **TODO: wrap the above into `download_data.sh` file**

4. Log into Azure Container Registry  
   We use Azure Container Registry (ACR) to store and manage Docker images. You can log into the ACR by
   ```bash
   docker login --username tsperf --password <ACR Access Key> tsperf.azurecr.io
   ```
   If want to execute docker commands without sudo as a non-root user, you need to create a Unix group and add users to it by following the instructions [here](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

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
   This will generate a `submission.xls` file under the `/benchmarks/submission1` folder. Then, you can evaluate the forecasting results by
   ```bash
   python3 energy_load/problem1/common/evaluate.py 'energy_load/problem1/benchmarks/submission1/submission.xls'
   ```
   The above command will output the evaluation metric of this submission.


# Instructions to submit new benchmark implementations
**TODO: Revise this section when we have a draft reference implementation**

1. Create a new git branch
2. Create a new numbered submission folder under ./GEFCom2017-D_Prob_MT_hourly/. Pick the next number available
3. Download and preprocess the problem dataset as above
4. To submit your solution you must create a script (in any language) that includes all code necessary to train your model and produce predictions for all forecasted periods in the format:

    * The file format should be *.xls;
    * The file name should be "submission.xls".
    * The file should include 10 worksheets, named as CT, ME, NEMASSBOST, NH, RI, SEMASS, VT, WCMASS, MASS, TOTAL. Please arrange the worksheets in the same order as listed above.
    * In each worksheet, the first two columns should be date and hour, respectively, in chronological order.
    * The 3rd to the 11th columns should be Q10, Q20, ... to Q90.

    The template is [HERE](https://www.dropbox.com/s/ksfiykyfqzmh3ph/TrackInitialRoundNumber-TeamName.xls?dl=0). You should replace the date column to reflect the forecast period in each round.

    The python helper function common/serve_folds.py can be used to serve up successive training and test period folds (see [./benchmarks/submission1/train_score.py](./benchmarks/submission1/train_score.py) for an example of how to use this). You are not obliged to use this function in your submission. You do not have to retrain or retune your model after each successive fold if it does not benefit your model performance. Forecasts **must not** be generated from models that have been trained on data of the forecast period or later. Submission code will be inspected to enforce this.

5. Once you have generated your submission file, you can evaluate the model's performance with
    ```
    python energy_load/problem1/common/evaluate.py \
    energy_load/problem1/benchmarks/<submission_dir>/submission.xls
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

# Leaderboard

| **Submission** | **Model description** | **Pinball Loss** |
| -------------- | --------------------- | --------- |
|  | |  |
