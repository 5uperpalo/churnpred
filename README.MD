## Additional data:

### GDPP:
https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.CD

### Big Mac index:
https://github.com/TheEconomist/big-mac-data

iso_codes: 
* not for free https://www.iso.org/publication/PUB500001.html
* updated in Jan 26; 1.3k github stars; good enough: https://github.com/stefangabos/world_countries/

###  Installation

Install using pip:

```bash
pip install churn_pred
```

Or install directly from github

```bash
pip install git+https://github.com/5uperpalo/churn_pred.git
```

#### Developer Install

```bash
# Clone the repository
git clone https://github.com/5uperpalo/churn_pred.git
cd pytorch-widedeep

# Install in dev mode
pip install -e .
```

#### Locally

```bash
pip install .
# or
pip install git+ssh://git@github.com/5uperpalo/churn_pred.git
```

### Quick start


```python
```

### Contribution

#### Code
Before pushing a code or making a pull request please run codestyle checks and tests
```bash
./code_style.sh
pytest --doctest-modules churn_pred --cov-report xml --cov-report term --disable-pytest-warnings --cov=churn_pred tests/
```