# <img src="docs/sources/assets/images/logo.png" style="height:2em; vertical-align: middle;">
Customer churn predictor for an assignment below.

Documentation: https://5uperpalo.github.io/churnpred/

```
In this assignment you're tasked with developing a machine learning solution for churn prediction to identify which customers are likely to leave a service (column "Exited" in the attached dataset). This assignment is meant to assess

- analytical skills and reasoning
- design and modelling choices, e.g. choices with respect to measuring model performance
- coding skills, e.g. modularity, readability, reproducibility, any other best practices in software development

Please note that multiple solutions may exist and we do not expect a production ready solution, though any reflections on how you may wish to productionalise your solution are welcome. You are free to choose the medium (e.g., notebooks, python scripts). 

Additional explanation of independent variables:

NumberOfProducts - the number of accounts and bank-affiliated products 
HasCreditCard - whether a customer has a credit card
CustomerFeedback - latest customer feedback, if available
```

## Solution

Please see the Notebooks section. The notebooks are sorted from 0 to 5. Notebooks start with gathering auxiliary data that I could extract from the provided dataset, e.g. 'country origin of the surname'. This is followed by Exploratory Data Analysis of features and target in the notebooks 2, 3. In the notebook 4, I presented a `Trainer` object that handles training an hyperparameter search of the model. In the notebook 5 I made a quick analysis of the model and it's predictions using SHAP values.

The final solution uses LightGBM, a GBM model of my choice. I chose GBM as 4 out of top 5 models in H2O AutoML were GBMs.


### Additional work note mentioning

In notebook `00_auxiliary_features_surname_origin_country_classification.ipynb` I adjusted(copy/paste+adjust) a BERT model for surname origin prediction. Due to lack of time I could not gather additional data that would help with model training, but I left some ideas in the notebook.

The solution was tested in a virtual machine, spawned from `jupyter/datascience-notebook:python-3.10` image in Zero-to-JupyterHub solution. As the bare metal server with GPU was down in the kubernetes, I had to do additional troubleshooting and fixing.

The code is easily extendable to `multiclass`, `regression` and `quantile_regression` tasks.

##  Installation

The code was tested on 
### Install using pip directly from github:

```bash
pip install git+https://github.com/5uperpalo/ecovadis_assignment.git
```

### Locally

```bash
git clone https://github.com/5uperpalo/ecovadis_assignment.git
cd ecovadis_assignment
pip install .
```

##  Documentation

```bash
# to build locally
cd docs
pip install -r requirements.txt
mkdocs build  --clean
# to push to github pages
mkdocs gh-deploy
# if you want to run webserver locally
mkdocs serve
```

## Code quality
Before pushing a code or making a pull request please run codestyle checks and tests
```bash
./code_style.sh
pytest --doctest-modules churn_pred --cov-report xml --cov-report term --disable-pytest-warnings --cov=churn_pred tests/
```