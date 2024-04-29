# sort imports
isort --quiet churn_pred tests notebooks
# Black code style
black churn_pred 
black tests 
black notebooks
# flake8 standards
flake8 . --max-complexity=10 --max-line-length=127 --ignore=E203,E266,E501,E722,E721,F401,F403,F405,W503,C901,F811
# mypy
mypy churn_pred --ignore-missing-imports --no-strict-optional