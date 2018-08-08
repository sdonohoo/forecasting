
## Structure of Repository

We use Git repo to maintain the source code and relevant files. The repository has three hierarchies: use case, problem, and benchmark implementation. A *use case* is a class of problems that focus on the forecasting of a certain quantity in a specify business domain. A *problem* is an explicitly defined forecasting problem based on a given dataset. A *benchmark implementation* is the implementation of a forecasting method that solves a specific problem. The top-level directory `/TSPerf` consists of folders for all the existing use cases, a folder storing common utility functions, a folder storing internal docs, and a Markdown file describing the time series benchmarking framework. 

* Use case folders: Each such folder is named after a specific use case and preserves the scripts of the benchmarks for every problem of this use case. Currently we have a folder `/TSPerf/energy_load` for the energy load forecasting use case. We plan to include another use case for retail demand forecasting in the initial release of TSPerf. 

  Under each use case folder, we have subfolders for different problems and a Markdown file listing all problems of this use case. For example, `/TSPerf/energy_load/GEFCom2017-D_Prob_MT_hourly` contains all the submissions for a probabilistic forecasting problem defined upon GEFCom2017-D dataset and `/TSPerf/energy_load/problem1` includes all the submissions for a point forecasting problem defined upon GEFCom2014 dataset. In addition, `/TSPerf/energy_load/README.md` summarizes all the problems of the energy load forecasting use case. 

  Under each problem folder, there are a subfolder containing source code of all the submissions, a subfolder storing common utility functions, and a Markdown file specifying the problem. The description of each item under the problem folder is as follows

    * `/benchmarks` folder: This folder contains multiple subfolders with each subfolder including all the necessary scripts and the submission form for reproducing a certain benchmark implementation. For instance, `/submission1` folder under `/TSPerf/energy_load/problem1/benchmarks` includes all the scripts of submission1 which is a reference benchmark implementation.
    

    * `/common` folder: As an example, `/TSPerf/energy_load/problem1/common` contains the modules that could be commonly used for problem1, such as Python modules that download the data, prepare training and scoring data, and evaluate performance of the benchmark. 

    * `/README.md`: For instance, `/TSPerf/energy_load/problem1/README.md` describes problem1 in detail and provides guidance about how to create submission of a benchmark and review the submission. 

* `/TSPerf/common` folder: This folder has the modules that could be used across different use cases, such as Python modules which compute the evaluation metrics of the forecasting results.

* `/TSPerf/internal_docs` folder: This folder contains the internal documents that we create during the development of TSPerf. 

* `/TSPerf/README.md` file: This Markdown file describes the TSPerf framework in general. It introduces the goal and vision, specifies the use cases and problems, as well as provides guidances for benchmark implementation, benchmark submission, and reviewing of the submissions. 

