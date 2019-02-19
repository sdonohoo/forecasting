# Implementation submission form

## Submission information

**Submission date**: 01/14/2018

**Benchmark name:** GEFCom2017_D_Prob_MT_hourly

**Submitter(s):** Dmitry Pechyoni

**Submitter(s) email:** dmpechyo@microsoft.com

**Submission name:** Quantile Random Forest

**Submission path:** energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/qrf


## Implementation description

### Modelling approach

In this submission, we implement a quantile random forest model using the `scikit-garden` package in Python.

### Feature engineering

The following features are used:  
**Basic temporal features**: hour of day, day of week, day of month, time of the year (normalized to range [0,1]), week of the year, month of the year  
**RecentLoad**: moving average of load values of the same day of
week and same hour of day of at the window of 4 weeks. We use 8 moving windows, the first one at weeks 10-13 before forecasting week, the last one is at weeks 17-20 before forecasting week. Each window generates a separate RecentLoad feature.  
**RecentDryBulb**:  moving average of Dry Bulb values of the same day of
week and same hour of day of at the window of 4 weeks. We use 8 moving windows, the first one at weeks 9-12 before forecasting week, the last one is at weeks 16-19 before forecasting week. Each window generates a separate RecentDryBulb feature.  
**RecentDewPnt**:  moving average of Dew Point values of the same day of
week and same hour of day of at the window of 4 weeks. We use 8 windows, the first one at weeks 9-12 before forecasting week, the last one is at weeks 16-19 before forecasting week. Each window generates a separate RecentDewPnt feature.  
**Daily Fourier Series features**: sine and cosine of the hour of the day, with harmonics 1 and 2. Altogether we generate 4 such features.  
**Weekly Fourier Series features**: sine and cosine of the day of the week, with harmonics 1, 2 and 3. Altogether we generate 6 such features.  
**Annual Fourier Series features**:  sine and cosine of the day of the year, with harmonics 1, 2 and 3. Altogether we generate 6 such features.

### Model tuning

We chose hyperparameter values that minimize average pinball loss over validation folds.
We used 2 validation time frames, the first one in January-April 2015, the second one at the same months in 2016. Each validation timeframe was partitioned into 6 folds, each one spanning entire month. The training set of each fold ends one or two months before the first date of validation fold.

### Description of implementation scripts

* `feature_engineering.py`: Python script for computing features and generating feature files.
* `train_score.py`: Python script that trains Quantile Random Forest models and predicts on each round of test data.
* `train_score_vm.sh`: Bash script that runs `feature_engineering.py`and `train_score.py` five times to generate five submission files and measure model running time.

### Steps to reproduce results

0. Follow the instructions [here](#resource-deployment-instructions) to provision a Linux Data Science Virtual Machine and log into it.

1. Clone the Forecasting repo to the home directory of your machine

    ```bash
    cd ~
    git clone https://github.com/Microsoft/Forecasting.git
    ```
  Use one of the following options to securely connect to the Git repo:
  * [Personal Access Tokens](https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/)  
  For this method, the clone command becomes
    ```bash
    git clone https://<username>:<personal access token>@github.com/Microsoft/Forecasting.git
    ```
  * [Git Credential Managers](https://github.com/Microsoft/Git-Credential-Manager-for-Windows)
  * [Authenticate with SSH](https://help.github.com/articles/connecting-to-github-with-ssh/)


2. Create a conda environment for running the scripts of data downloading, data preparation, and result evaluation.   
To do this, you need to check if conda has been installed by runnning command `conda -V`. If it is installed, you will see the conda version in the terminal. Otherwise, please follow the instructions [here](https://conda.io/docs/user-guide/install/linux.html) to install conda.  
Then, you can go to `~/Forecasting` directory in the VM and create a conda environment named `tsperf` by running

   ```bash
   cd ~/Forecasting
   conda env create --file ./common/conda_dependencies.yml
   ```

3. Download and extract data **on the VM**.

    ```bash
    source activate tsperf
    python energy_load/GEFCom2017_D_Prob_MT_hourly/common/download_data.py
    python energy_load/GEFCom2017_D_Prob_MT_hourly/common/extract_data.py
    ```

4. Prepare Docker container for model training and predicting.  
   4.1 Log into Azure Container Registry (ACR)

   ```bash
   sudo docker login --username tsperf --password <ACR Access Key> tsperf.azurecr.io
   ```

   The `<ACR Acccess Key>` can be found [here](https://github.com/Microsoft/Forecasting/blob/master/common/key.txt).   
   If want to execute docker commands without
   sudo as a non-root user, you need to create a
   Unix group and add users to it by following the instructions
   [here](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

   4.2 Pull the Docker image from ACR to your VM

   ```bash
   sudo docker pull tsperf.azurecr.io/energy_load/gefcom2017_d_prob_mt_hourly/qrf_image:v1
   ```

5. Train and predict **within Docker container**  
  5.1 Start a Docker container from the image  

   ```bash
   sudo docker run -it -v ~/Forecasting:/Forecasting --name qrf_container tsperf.azurecr.io/energy_load/gefcom2017_d_prob_mt_hourly/qrf_image:v1
   ```

   Note that option `-v ~/Forecasting:/Forecasting` mounts the `~/Forecasting` folder (the one you cloned) to the container so that you can access the code and data on your VM within the container.

   5.2 Train and predict  

   ```
   source activate tsperf
   nohup bash /Forecasting/energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/qrf/train_score_vm.sh >& out.txt &
   ```
   The last command will take about 31 hours to complete. You can monitor its progress by checking out.txt file. Also during the run you can disconnect from VM. After reconnecting to VM, use the command  

   ```
   sudo docker exec -it qrf_container /bin/bash
   tail out.txt
   ```
   to connect to the running container and check the status of the run.  
   After generating the forecast results, you can exit the Docker container by command `exit`.   

6. Model evaluation **on the VM**

    ```bash
    source activate tsperf
    cd ~/Forecasting
    bash ./common/evaluate submissions/qrf energy_load/GEFCom2017_D_Prob_MT_hourly
    ```

## Implementation resources

**Platform:** Azure Cloud  
**Resource location:** East US region   
**Hardware:** F72s v2 (72 vcpus, 144 GB memory) Ubuntu Linux VM  
**Data storage:** Standard SSD  
**Docker image:** tsperf.azurecr.io/energy_load/gefcom2017_d_prob_mt_hourly/qrf_image  

**Key packages/dependencies:**
  * Python
    - python==3.6    
    - numpy=1.15.1
    - pandas=0.23.4
    - xlrd=1.1.0
    - urllib3=1.21.1
    - scikit-garden=0.1.3
    - joblib=0.12.5  

## Resource deployment instructions
Please follow the instructions below to deploy the Linux DSVM.
  - Create an Azure account and log into [Azure portal](portal.azure.com/)
  - Refer to the steps [here](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) to deploy a *Data Science Virtual Machine for Linux (Ubuntu)*.  Select *F72s_v3* as the virtual machine size.   


## Implementation evaluation
**Quality:**  

* Pinball loss run 1: 76.48

* Pinball loss run 2: 76.49

* Pinball loss run 3: 76.43

* Pinball loss run 4: 76.47

* Pinball loss run 5: 76.6

* Median Pinball loss: 76.48

**Time:**

* Run time 1: 22289 seconds

* Run time 2: 22493 seconds

* Run time 3: 22859 seconds

* Run time 4: 22709 seconds

* Run time 5: 23197 seconds

* Median run time: 22709 seconds (6.3 hours)

**Cost:**  
The hourly cost of the F72s v2 Ubuntu Linux VM in East US Azure region is 3.045 USD, based on the price at the submission date.   
Thus, the total cost is 22709/3600 * 3.045 = 19.21 USD.

**Average relative improvement (in %) over GEFCom2017 benchmark model**  (measured over the first run)  
Round 1: 16.84  
Round 2: 14.98  
Round 3: 12.08  
Round 4: 14.97  
Round 5: 16.16  
Round 6: -2.52  

**Ranking in the qualifying round of GEFCom2017 competition**  
3
