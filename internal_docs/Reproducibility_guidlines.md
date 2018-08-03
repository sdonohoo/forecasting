## Reproducibility Guidelines
### System and framework availability
This section is aligned with MLPerf.  
If you are using a publicly available system or framework, you must use publicly available and widely-used used versions of the system or framework.  
If you are using an experimental framework or system, you must make the system and framework you use available upon request for replication.
### Benchmark implementation source code
This section is aligned with MLPerf.  
Source code used for the benchmark implementations must be open-sourced under a license that permits a commercial entity to freely use the implementation for benchmarking. The code must be available as long as the results are actively used.
### Environment setup
1. Parallel/distributed computation environment setup  
If you are using multiple machines for parallel/distributed computation, you must provide a script for automatically creating the cluster (preferred) or instructions for manual creation.
2. Virtual machine or Docker image setup  
You need to provide instructions for setting up the implementation system from a plain VM, or a Docker file/image for creating the container needed to execute the implementation.
3. Virtual environment setup  
If your implementation is light-weight and does not have any system dependency, a YAML file for creating a conda environment is also acceptable.
4. Framework and package version report  
The submitter needs to submit a report summarizing all the framework and package versions used for producing the reported result. This is to prevent the newer version of a framework or package significantly changing the implementation result.

### Non-determinism restrictions
This section is aligned with MLPerf. Some more detailed instructions are added.  
The following forms of non-determinism are acceptable in MLPerf.
- Floating point operation order. For example, certain functions in cuDNN do not guarantee reproducibility across runs.

- Random initialization of the weights and/or biases.

- Random traversal of the inputs.  

In order to avoid any other sources of non-determinisms, we recommend setting random seeds whenever a package/framework provides a function for setting random seed, e.g. numpy.random.seed(), random.seed(), tf.set_random_seed().  
The submitter needs to run the benchmark implementation five times using the integer random number generator seeds 1 through 5 and report all five results.  The variance of the five run results should be reasonable, otherwise, it's an indicator of instability of the implementation. The median of the five results is reported as the performance of the submitted implementation. 

### Hyperparameter tuning
Instructions for hyperparameter tuning is optional. However, it's **highly recommended** to provide details of your hyperparameter tuning process, which will make it easier to adopt an implementation to a new dataset.
