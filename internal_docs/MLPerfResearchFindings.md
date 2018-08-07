### If reference implementations in MLPerf include tuning of hyperparameters
None of the reference implementations included hyper parameter tuning. Most of
the time, the parameters are set in the bash script that calls the main python
script. Sometimes, the parameters are hard coded in the python script.

I think hyperparameter tuning is an important aspect of adopting a submitted
method to a different dataset, thus we should recommend the participants to
submit instructions for hyperparameter tuning, and we should include in our
reference implementations. As we discussed earlier, it will be hard to include
the time for hyperparameter tuning in the measured computation time, as
someone could set the parameter searching space to a very specific range to
reduce tuning time.

### If reference implementations in MLPerf use training/validation or training/validation/test splits  
Except for the speech_recognition and reinforcement benchmarks, all the other
benchmarks only has training and test data, mostly pre-determined by the data
sources.

### How MLPerf handles reproducibility issue
All the reference implementations provided detailed instructions for setting
up machine and docker (when needed), bash scripts for executing python scripts.

All the reference implementations set random seed whenever possible, including
python random seed, numpy random seed, tensorflow random seed, and pytorch
random seed. However, due to unfamiliarity with these frameworks, it's not clear
whether the results are deterministic even with these random seeds set. For example,
in an implementation using caffee2, there is a comment in the code saying
"Note that while we set the numpy random seed network training will not be
deterministic in general. There are sources of non-determinism that cannot
be removed with a reasonble execution-speed tradeoff (such as certain
non-deterministic cudnn functions)."

According to the the User Guide, The only forms of acceptable nondeterminism are:
- Floating point operation order
- Random initialization of the weights and/or biases
- Random traversal of the inputs
- Reinforcement learning exploration decisions

The MLPerf User Guide also mentioned "All random numbers must be drawn from the
framework’s stock random number generator. The
random number generator seed must entirely determine its output sequence"

Here is how reference result and benchmark results are defined in the MLPerfUser
Guide.
- A reference result is the median of five run results provided by the MLPerf organization for each
reference implementation.
- A benchmark result is the median of five run results normalized to the reference result for that
benchmark. Normalization.

It seems like a valid submission is always expected to meet the minimum required accuracy in every run.
