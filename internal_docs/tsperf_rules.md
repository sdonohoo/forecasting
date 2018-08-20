# TSPerf Rules

## Introduction

Vision, goals (Ilan - 3)

## Framework

Framework goals

### Definitions 
We adopt several definitions from [MLPerf](https://github.com/mlperf/policies/blob/master/rules.adoc) 
and also add a number of new ones:  

A *use case* is a broad family of abstract problems in a specific vertical. Examples include energy 
demand forecasting, retail sales forecasting, medical image classification, 

A *benchmark* is a specific abstract problem that can be solved using ML by training a model based on a 
specific dataset.

*Benchmark rules* is a set of rules for implementing a benchmark to produce a class of comparable 
results, such as training and test set partition, performance evaluation metric and process.

A *system* consists of a defined set of cloud hardware resources and services such as processors, 
memories, disks, clusters and interconnect. It also includes specific versions of all software such as 
operating system, compilers, libraries, and drivers that significantly influences the running time, 
excluding the ML framework.

A *framework* is a specific version of a software library or set of related libraries, possibly with 
associated offline compiler, for training ML models using a system. Examples include specific versions 
of Caffe2, MXNet, PaddlePaddle, pyTorch, or TensorFlow.

A *benchmark implementation* is an implementation of a benchmark in a particular framework by a user 
under the benchmark rules.

*Benchmark quality* is performance metric value of a benchmark implementation, measured according to 
benchmark rules.

A *run* is a complete execution of a benchmark implementation on a system, training a model from 
initialization to the specified  quality target.

A *run result* is the wallclock time and cost required for a run.

A *benchmark result*  is the median of five run results.

A *benchmark submission* is a source code of benchmark implementation, along with declared benchmark 
quality and  benchmark result.

A *validated benchmark submission* is a benchmark submission that passed review process.

A *benchmark leaderboard* is a table of validated benchmark submissions.

A *benchmark reference implementation* is a specific validated benchmark submission chosen from the 
leaderboard. 

The following diagram summarizes the relations betwwen different terms:

<img src="./images/definitions.png" alt="drawing" heigh="300px" width="600px"/>

### Structure and hierarchy of documents
Ilan

### Structure of repository

We use Git repo to maintain the source code and relevant files. The repository has three levels of folders: use case, benchmark, and benchmark implementation.
The top-level directory `/TSPerf` consists of folders for all the existing use cases, a folder storing common utility scripts, a folder storing internal 
docs, and a Markdown file describing the time series benchmarking framework. 

* Use case folders: Each such folder is named after a specific use case and contains scripts of the implementations/submissions for every benchmark of this 
use case. Currently we have a folder `/TSPerf/energy_load` for the energy load forecasting use case and another folder `/TSPerf/retail_sales` for the 
retail sales forecasting use case. 

  Under each use case folder, we have subfolders for different benchmarks and a Markdown file listing all benchmarks of this use case. For example, 
  `/TSPerf/energy_load/GEFCom2017-D_Prob_MT_hourly` contains all the submissions for a probabilistic forecasting problem defined upon GEFCom2017-D dataset 
  and `/TSPerf/energy_load/GEFCom2014_Pt_1Month_Hourly` includes all the submissions for a point forecasting problem defined upon GEFCom2014 dataset. In 
  addition, `/TSPerf/energy_load/README.md` summarizes all the benchmarks of the energy load forecasting use case. 

  Under each benchmark folder, there are a subfolder containing source code of all reference implementations, a subfolder containing source code of all 
  submissions, a subfolder storing common utility scripts, and a Markdown file specifying the benchmark. The description of each item under the benchmark 
  folder is as follows

    * `/reference` folder: This folder contains all the necessary scripts and the submission form for reproducing the reference implementation. For 
    instance, `/TSPerf/energy_load/GEFCom2014_Pt_1Month_Hourly/reference` includes the required submission files of the reference implementation for 
    GEFCom2014_Pt_1Month_Hourly.

    * `/submissions` folder: This folder contains multiple subfolders with each subfolder including all the necessary scripts and 
    the submission form for reproducing a certain submission. For instance, `/submission1` folder under 
    `/TSPerf/energy_load/GEFCom2014_Pt_1Month_Hourly/submissions` includes the required submission files of submission1.
    

    * `/common` folder: This folder includes utility scripts for a benchmark. As an example, `/TSPerf/energy_load/GEFCom2014_Pt_1Month_Hourly/common` 
    contains the scripts that could be commonly used for GEFCom2014_Pt_1Month_Hourly, such as Python scripts that download the data, prepare training and 
    scoring data, and evaluate performance of the benchmark implementation. 

    * `/README.md`: This Markdown file provides detailed instructions about a certain benchmark. For instance, 
    `/TSPerf/energy_load/GEFCom2014_Pt_1Month_Hourly/README.md` describes benchmark and provides benchmark-specific guidance for evaluating model 
    performance and creating benchmark submission. 

* `/TSPerf/common` folder: This folder has the scripts that could be used across different use cases, such as Python scripts which compute the evaluation 
metrics of the forecasting results.

* `/TSPerf/internal_docs` folder: This folder contains the internal documents that we create during the development of TSPerf. 

* `/TSPerf/README.md` file: This Markdown file describes the TSPerf framework in general. It introduces the goal and vision, specifies the use cases and 
benchmarks, as well as provides guidances for benchmark implementation, benchmark submission, and reviewing of the submissions. 

## Benchmarks  

| **Benchmark problem** | **Benchmark directory** |  
| --------------------- | -------------------- |  
| Probabilistic electricity load forecasting | TSPerf\energy_load\GEFCom2017-D_Prob_MT_Hourly |
| Retail sales forecasting | TsPerf\retail_sales\OrangeJuice_Pt_3Weeks_Weekly |

### Probabilistic electricity load forecasting

Probabilistic load forecasting (PLF) has become increasingly important in
power systems planning and operations in recent years. The applications of PLF
include energy production planning, reliability analysis, probabilistic price
forecasting, etc.
The task of this benchmark is to generate probabilistic forecasting of
electricity load on the GEFCom2017 competition qualifying match data. We use
about 6 years of data for training. Forecast is done at 6 time points, with
horizon of 1 ~ 2 months and granularity of 1 hour. The forecasts should be in
the form of 9 quantiles, i.e. the 10th, 20th, ... 90th percentiles, following
the format of the provided template file.  There are 10 time series (zones) to
forecast, including the 8 ISO New England zones, the Massachusetts (sum of
three zones under Massachusetts), and the total (sum of the first 8 zones).
The quality metric of this benchmark is the Pinball loss function.

### Retail sales forecasting


Sales forecasting is a key task for the management of retail stores. With the projection of future sales, store managers will be able to optimize 
the inventory based on their business goals. This will generate more profitable order fulfillment and reduce the inventory cost. 

The task of this benchmark is to forecast orange juice sales of different brands for multiple stores with the Orange Juice dataset from R package 
`bayesm`. The forecast type is point forecasting. Forecast is done at 12 time points, with horizon of 3 weeks and granularity of 1 week. The forecasts should  
include the predicted sales during the target periods for each brand and each store. The output file of the forecast results should follow the format of the 
provided template file. There are in total 913 time series to forecast. The quality metric of this benchmark is the mean average percentage error. 

## Development of benchmark implementation

### Availables Docker images

We recommend to use Docker images for the reproduciblility of the submissions. In TSPerf, we provide a basic Docker image to speed up the process of new benchmark implementation, namely tsperf.azurecr.io/common/image:v1 in tsperf Azure Container Registry (ACR). This image contains basic configurations of the system and a few commonly used packages. One can directly use the basic image by pulling it from the ACR or modify it for their own benchmark implementations. 

Under `/TSPerf/common` folder, there are a Dockerfile and requirements.txt file used for creating the basic image. The Dockerfile contains the main configuration steps and requirements.txt includes the necessary Python packages. By modifying these files, one can easily create their own Docker images and host them in ACR or other venues such as Docker Hub.

## Guideline for measuring performance

Each benchmark result is the median of five run results produced using the integer random number generator seeds 1 through 5. All five run results must also be reported. The following measurements should be included:
  * quality of the model
  * running time
  * cloud cost 

### Quality of the Model

The quality of the model is measured by a certain evaluation metric e.g. MAPE. Please use common utility script `evaluate.py` to get the benchmark quality value in each run
```bash
python <benchmark directory>/common/evaluate.py <submission directory>/submission_seed_<seed value>.csv
``` 

### Running Time

The wallclock running time of each run should be measured by 
```bash
time -p python <submission directory>/train_score.py
```

### Cloud Cost

Include the total cost of obtaining the median run result using fixed prices for the general public at the time the result is collected. Do not use spot 
pricing. If you use Azure, you can estimate the costs for Azure products using this [online pricing calculator](https://azure.microsoft.com/en-us/pricing/calculator/).  

## Submission

### Guideline for submitting reproduction instructions

#### System and framework availability
This section is aligned with MLPerf.  
If you are using a publicly available system or framework, you must use publicly available and widely-used used versions of the system or framework.  
If you are using an experimental framework or system, you must make the system and framework you use available upon request for replication.

#### Benchmark implementation source code
This section is aligned with MLPerf.  
Source code used for the benchmark implementations must be open-sourced under a license that permits a commercial entity to freely use the implementation for benchmarking. The code must be available as long as the results are actively used.

#### Environment setup
1. Parallel/distributed computation environment setup  
If you are using multiple machines for parallel/distributed computation, you must provide a script for automatically creating the cluster (preferred) or instructions for manual creation.
2. Virtual machine or Docker image setup  
You need to provide instructions for setting up the implementation system from a plain VM, or a Docker file/image for creating the container needed to execute the implementation.
3. Virtual environment setup  
If your implementation is light-weight and does not have any system dependency, a YAML file for creating a conda environment is also acceptable.
4. Framework and package version report  
The submitter needs to submit a report summarizing all the framework and package versions used for producing the reported result. This is to prevent the newer version of a framework or package significantly changing the implementation result.

#### Non-determinism restrictions
This section is aligned with MLPerf. Some more detailed instructions are added.  
The following forms of non-determinism are acceptable in MLPerf.
- Floating point operation order. For example, certain functions in cuDNN do not guarantee reproducibility across runs.

- Random initialization of the weights and/or biases.

- Random traversal of the inputs.  

In order to avoid any other sources of non-determinisms, we recommend setting random seeds whenever a package/framework provides a function for setting random seed, e.g. numpy.random.seed(), random.seed(), tf.set_random_seed().  
The submitter needs to run the benchmark implementation five times using the integer random number generator seeds 1 through 5 and report all five results.  The variance of the five run results should be reasonable, otherwise, it's an indicator of instability of the implementation. The median of the five results is reported as the performance of the submitted implementation. 

#### Hyperparameter tuning
Instructions for hyperparameter tuning are optional. However, it's **highly recommended** to provide details of your hyperparameter tuning process, which will make it easier to adopt an implementation to a new dataset.

### Guideline for submitting the code
Guidance for submitting the code  
Chenhui, Ilan

## Review of submissions
The goal of the review is to validate the declared

* quality of the model 
* running time
* cloud cost  

We will explain below how to validate these quantities. Additionally the reviewer should check that the 
evaluation of the quality of the model is done using a standard `evaluate.py` script and that the code

* does not use test data for training
* has a good quality 
* is well documented

We do not have specific guidelines for checking these items. Reviewer should use his/her own judgement 
to decide if there is no test data leakage and if the code or documentation need improvement.

Reviewer should set up execution environment before running benchmark implementation. Initially the 
reviewer needs to complete the following three steps:

0. Verify that the submission has README.md file with  
    * name of the branch with submission code
    * benchmark path, for example /TSPerf/energy_load/problem1
    * path to submission directory, for example  /TSPerf/energy_load/problem1/benchmarks/submission1
    * instructions for provisioning the system (e.g. DSVM, Batch AI)
    * name of Docker image stored in tsperf registry, for example 
    tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1

In the following sections all occurences of "README file" refer to README file in the submission, unless 
specified otherwise.

The next steps depend on the system and are described in the following two subsections.

### Standalone VM

1. Follow the instructions in the README file and provision the virtual machine that was used to generate 
benchmark results. Then log into the provisioned VM.

2. Choose submission branch and clone the Github repo to your machine:

        git clone https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_git/TSPerf
        git checkout <branch name>

3. Download the data using the following commands

        python <benchmark path>/common/get_data.py

    where \<benchmark path\> is a root benchmark directory, for example energy_load/problem1

4. Log into Azure Container Registry:
   
       docker login --username tsperf --password <ACR Access Key> tsperf.azurecr.io
   
   If want to execute docker commands without sudo as a non-root user, you need to create a Unix group and 
   add users to it by following the instructions 
   [here](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

5. Pull a Docker image from ACR, using image name that is specified in README file:   
      
       docker pull <image name>

6. Choose a name for a new Docker container and create it using command:   
   
       docker run -it -v ~/TSPerf:/TSPerf --name <container name> <image name>
   
   Note that you need to mount `/TSPerf` folder (the one you cloned) to the container so that you will 
   have access to the source code in the container. 

7. Inside Docker container, run the following command:  

       source <benchmark directory>/common/train_score_vm <submission directory> 

   This will generate 5 `submission_seed_<seed number>.csv` files in the submission directory, where \<seed number\> 
   is between 1 and 5. This command will also output 5 running times of train_score.py. The median of the times 
   reported in rows starting with 'real' should be compared against the wallclock time declared in benchmark 
   submission.
   
8. Evaluate the benchmark quality by running

       source <benchmark directory>/common/evaluate <submission directory> <benchmark directory>

    This command will output 5 benchmark quality values (e.g. MAPEs). Their median should be compared against the 
    benchmark quality declared in benchmark submission.

### Batch AI

1. Provision Linux Data Science VM with DS4v2 configuration. 

2. Choose submission branch and clone the Github repo to your machine:

        git clone https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_git/TSPerf
        git checkout <branch name>

3. Download the data using the following commands

        python <benchmark path>/common/get_data.py

    where \<benchmark path\> is a root benchmark directory, for example energy_load/problem1

4. Follow the instructions in the README file and provision the resource group, Batch AI workspace, Batch AI cluster 
and storage account with file share and blob storage. When creating resources, please record *resource group name*, 
*storage account name*, *storage account key* and *file share name*. You will need to provide these parameters in the 
later steps.

5. Upload <submission directory>/train_score.py script to file share account by following the instructions in README 
file.

6. Upload the dataset to blob storage by following the instructions in README file.

7. Create Batch AI experiment:

       az batchai experiment create -g <resource group name> -w <workspace name > -n <submission name>

   where resource group name and workspace name are the ones used in step 4. Submission name is rightmost directory in 
   submission path. 

8. Run 5 Batch AI jobs

        source  train_score_batchai <resource group name> <workspace name> <cluster name> <submission directory> 
        <storage account name>

    This command will create 5 `submission_seed_<seed number>.csv` files in the local directory, where \<seed number\> is 
    between 1 and 5. This command will also output 5 running times of Batch AI jobs. The median of these times should 
    be compared against the wallclock time declared in benchmark submission. 

9. Evaluate the benchmark quality by running

       source <benchmark directory>/common/evaluate <submission directory> <benchmark directory>

    This command will output 5 benchmark quality values (e.g. MAPEs). Their median should be compared against the 
    benchmark quality declared in the benchmark submission.

## Leaderboard
Each benchmark will have a separate leaderboards. All leaderboards will have the following columns:
* submission name
* URL of submission folder in VSTS
* benchmark quality (e.g. MAPE)
* running time
* cost
* system (e.g. DSVM)
* framework (e.g. Tensorflow)
* algorithm (e.g. LSTM)  

Each row will be a validated benchmark submission. Leaderboard will be updated by reviewer, after validating a benchmark 
submission. Every validated benchmark submission will be shown in the leaderboard. Since benchmark submission are measured 
by three parameters (quality, running time and cost), there will be no ranking between leaderboard entries.

## Selection of reference implementation

How reference implementation will be selected   
Dmitry
