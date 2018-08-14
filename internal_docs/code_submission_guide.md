## Guideline for Submitting Code and Results

New benchmark submissions to TSPerf should be made through pull requests in the Git repo by adding a completed submission form and supporting code in a new 
submission folder. 

The submission process depends on the system and is described in the following two subsections.

### Standalone VM

0. Provision a virtual machine and log into the provisioned VM. 

1. Clone the Git repo and create a new git branch
   ```bash
   git clone https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_git/TSPerf   
   git checkout -b <branch name>
   ```

2. Create a new submission folder under `<benchmark directory>/submissions` where `<benchmark directory>` is a root benchmark directory, e.g., 
`/LSTM_3layers` under `energy_load/<benchmark name>/submissions`. Please name your submission folder by following a similar naming convention.

3. Download the data using the following commands
   ```bash
   python <benchmark directory>/common/get_data.py
   ``` 

4. Implement a benchmark with the downloaded data to solve an associated problem defined by TSPerf. During the implementation, please use only the data 
specified in the problem description to train your model and evaluate model performance based on the specified testing data. Always check the common utility 
folder under `<benchmark directory>` to see if there is a module which creates the training and testing data for each forecast period, e.g. 
`energy_load/problem1/common/serve_folds.py`. Use this module as much as possible. Forecasts **must not** be generated from models that have been trained on 
data of the forecast period or later. Submission code will be inspected to enforce this.

5. To submit your solution you must create a script (in any language) that includes all code necessary to train your model and produce predictions for all 
forecasted periods in the required format. This script should be named as `train_score.*` with * indicating the file type. This script should accept an input argument which is an integer random number generator seed. For example, it should run as `python <submission directory>/train_score.py <seed value>` if it is in Python.  Moreover, the predictions should be stored in an Excel file named `submission_seed_<seed value>.csv`. Please include 
`train_score.*` script and `submission_seed_<seed value>.csv` files in your submission directory. 

6. Once you have generated your submission file, you can evaluate the model's performance with
   ```bash
   python <benchmark directory>/common/evaluate.py <submission directory>/submission_seed_<seed value>.csv 
   ```
   This command will output benchmark quality value (e.g. MAPE).

7. Report the median of five run results produced using the integer random number generator seeds 1 through 5. This can be done by running the model 
training and scoring script as follows
   ```bash
   python <submission directory>/train_score.py --seed RANDOM_SEED
   ```
   where RANDOM_SEED is an integer between 1 and 5. Evaluate the performance of each run with `evaluate.py` and compute the median.

8. Include other scripts that are necessary for reproducing the submitted benchmark results. For example, you should include a Dockerfile containing all 
dependencies for running your benchmark submission. The Dockerfile can point to a `.txt` file which contains a list of necessary packages. If you use Batch 
AI, please include a JSON file `job.json` that describes the Bath AI job. 

9. Create a Docker image and push it to the ACR   
   To create your Docker image, for example you can go to `/submissions/submission1/` folder and run the following command   
   ```bash
   docker build -t submission1_image .
   ```
   Then, you can push the image to ACR by executing
   ```bash
   docker tag submission1_image tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1
   docker push tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1
   ```
   Note that you will need to log into the ACR before publishing the image.


10. Include a submission form in the submission folder as README.md file. The submission form documents the submitter's information, method utlized in the 
benchmark implementation, information about the scripts, obtained results, and steps of reproducing the results. An example submission form can be found 
here (TODO: Add link). Specifically, it should include
    * name of the branch with submission code
    * benchmark path, e.g. `/TSPerf/energy_load/problem1`
    * path to submission directory, e.g. `/TSPerf/energy_load/problem1/submissions/submission1`
    * instructions for provisioning the system (e.g. DSVM, Batch AI)
    * name of Docker image stored in tsperf registry, for example
      tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1
    * benchmark quality values obtained with random seeds 1 through 5

11. Create pull request for review