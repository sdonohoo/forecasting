
## Structure of Repository

We use Git repo to maintain the source code and relevant files. The repository has three hierarchies: use case, benchmark, and benchmark implementation.
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

