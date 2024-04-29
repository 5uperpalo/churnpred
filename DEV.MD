## Carbon emission monitoring:
* https://mlco2.github.io/codecarbon/usage.html#
* https://github.com/sb-ai-lab/Eco2AI
* https://github.com/lfwa/carbontracker


## RAM usage profiling and Code execution timing

### RAM usage profiling
<details><summary>Max RAM used</summary>

To get max RAM used during executition of the function import the decorator and put it above it function to log max RAM used into specified logging file, e.g.:
**NOTE**: this applies only to functions and NOT classes or class methods. In case somebody figures out how to use it also on methods please let me know[Pavol Mulinka]. Posts relevant to the issue I was unable to debug:
* https://stackoverflow.com/questions/75409483/why-do-i-get-missing-required-argument-self-when-using-a-decorator-written-as
* https://stackoverflow.com/questions/16593246/how-to-use-memory-profiler-python-module-with-class-methods
```
from churn_pred.code_profiling import ram_usage

@ram_usage
def training(config: TrainingConfig, custom_params: CustomParameters):
```

</details>
<details><summary>Per line RAM usage analysis</summary>
To analyze RAM usage per line of the code import profile decorator, put it above the function you want to analyze:

```
from memory_profiler import profile

@profile
def training(config: TrainingConfig, custom_params: CustomParameters):
```

, run the code using memory_profile command line util, and analyze the RAM usage, e.g.:

```
(pltv_gmls) ubuntu@ip-172-31-10-63:~/repos/pltv-gmls$ python -m memory_profiler local_run.py

Filename: /home/ubuntu/repos/pltv-gmls/lgb_ltv/pipeline/training/main.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    53    230.8 MiB    230.8 MiB           1   @ram_usage
    54                                         @profile
    55                                         def training(config: TrainingConfig, custom_params: CustomParameters):
    56    230.9 MiB      0.1 MiB           1       data_config = DataConfig.load(config.data_config)
    57    230.9 MiB      0.0 MiB           1       hyperparameters = Hyperparameters.parse_obj(config.hyperparameters)
    58                                         
    59    687.5 MiB    456.6 MiB           1       queries, df_data = get_data(config, custom_params)
    60    687.5 MiB      0.0 MiB           1       (
```
</details>

### Code Execution timing
<details><summary>Timer decorator</summary>
Import and prepend the time decorator to log time(into specified log file) that it takes to execute analyzed function/class, e.g.:

```
from churn_pred.code_profiling import timer

@timer
def training(config: TrainingConfig, custom_params: CustomParameters):
```
</details>