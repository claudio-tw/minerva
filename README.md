# MINERVA

### MINE-based feature selection
Minerva is a feature selection tool 
based on 
neural estimation of mutual information 
between
features 
and 
targets.
A detailed explanation of 
our feature selection methodology is available in
the accompanying paper. 


## Installation
### Install from source
Checkout the repository and navigate to the root directory. Then, 

```
$ poetry install
```

### Run tests
After installation you can run the unit tests by doing:
```
$ poetry run pytest
```

## Run experiments

The repository collects several experiments of feature selection using Minerva. 
You can find them in the directory `experiments/`. 
You can use those scripts to reproduce our results. 
For example, you can:

1. Estimate mutual information between two normal random variables at different level of correlation:
```
$ python experiments/normalsmile.py
```
Tensorboard logs will be available at `tb_logs/normalsmile/`
The same experiment can be run through the notebook `notebooks/normalsmile.ipynb`.



2. Feature selection in a linear trasnformation setting:
```
$ python experiments/linear.py
```
Tensorboard logs will be available at `tb_logs/linear/`
The same experiment can be run through the notebook `notebooks/linear.ipynb`.



Moreover,
the experiments discussed in the paper
were run using the scripts
in `experiments/experiment_1`.
