# Assignment 

```
In this assignment you're tasked with developing a machine learning solution for churn prediction to identify which customers are likely to leave a service (column "Exited" in the attached dataset). This assignment is meant to assess

- analytical skills and reasoning
- design and modelling choices, e.g. choices with respect to measuring model performance
- coding skills, e.g. modularity, readability, reproducibility, any other best practices in software development

Please note that multiple solutions may exist and we do not expect a production ready solution, though any reflections on how you may wish to productionalise your solution are welcome. You are free to choose the medium (e.g., notebooks, python scripts). 

Please return your solution by May 2 (9am CET). Once returned, our colleagues with the HR department will schedule the final interview where you'll be asked to walk us through your solution and reflect on any decisions made.

Additional explanation of independent variables:

NumberOfProducts - the number of accounts and bank-affiliated products 
HasCreditCard - whether a customer has a credit card
CustomerFeedback - latest customer feedback, if available
```

## Additional data:

### GDPP:
https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.CD

### Big Mac index:
https://github.com/TheEconomist/big-mac-data

iso_codes: 
* not for free https://www.iso.org/publication/PUB500001.html
* updated in Jan 26; 1.3k github stars; good enough: https://github.com/stefangabos/world_countries/

##  Installation

The code was tested on 
### Install using pip directly from github:

```bash
pip install git+https://github.com/5uperpalo/churn_pred.git
```

### Locally

```bash
git clone https://github.com/5uperpalo/churn_pred.git
cd churn_pred
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