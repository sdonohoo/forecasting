# TSPerf Rules

## Introduction

Vision, goals (Ilan - 3)

## Framework

### Definitions 
We adopt several definitions from [MLPerf](https://github.com/mlperf/policies/blob/master/rules.adoc) and also add a number of new ones:

A *benchmark* is an abstract problem that can be solved using ML by training a model based on a specific dataset.

*Benchmark rules* is a set of rules for implementing a benchmark to produce a class of comparable results, such as training and test set partition, performance evaluation metric and process.

A *system* consists of a defined set of cloud hardware resources and services such as processors, memories, disks, clusters and interconnect. It also includes specific versions of all software such as operating system, compilers, libraries, and drivers that significantly influences the running time, excluding the ML framework.

A *framework* is a specific version of a software library or set of related libraries, possibly with associated offline compiler, for training ML models using a system. Examples include specific versions of Caffe2, MXNet, PaddlePaddle, pyTorch, or TensorFlow.

A *benchmark implementation* is an implementation of a benchmark in a particular framework by a user under the benchmark rules.

*Benchmark quality* is performance metric value of a benchmark implementation, measured according to benchmark rules.

A *run* is a complete execution of a benchmark implementation on a system, training a model from initialization to the specified  quality target.

A *run result* is the wallclock time and cost required for a run.

A *benchmark result*  is the median of five run results.

A *benchmark submission* is a source code of benchmark implementation, along with declared benchmark quality and  benchmark result.

A *validated benchmark submission* is a benchmark submission that passed review process.

A *benchmark leaderboard* is a table of validated benchmark submissions.

A *benchmark reference implementation* is a specific validated benchmark submission chosen from the leaderboard. 

The following diagram summarizes the relations betwwen different terms: 

<img src="images/definitions.png" alt="drawing" heigh="300px" width="600px"/>

### Structure and hierarchy of documents 
Ilan

### Structure of repository 
Chenhui

## Problems
training, test sets, what should be predicted, metrics

### Energy consumption forecasting  
Hong

### Retail sales forecasting 
Chenhui

## Model development

### Availables Docker images
Description of the available Docker images  
Chenhui

### Guideline for measuring performance
Guideline for measuring performance  
Chenhui 

## Submission

### Guideline for submitting reproduction instructions
Guidance for submitting reproduction instructions  
Hong, Ilan  

### Guideline for submitting the code
Guidance for submitting the code  
Chenhui, Ilan 

## Review of submissions



## Leaderboard

Structure of the leaderboard  
Dmitry

## Selection of reference implementation

How reference implementation will be selected   
Dmitry