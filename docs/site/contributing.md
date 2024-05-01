# Code
Before making a pushing a code and making a pull request please run codestyle checks and tests

```bash
./code_style.sh
pytest --doctest-modules churn_pred --cov-report xml --cov-report term --disable-pytest-warnings --cov=eda tests/
```

# Documentation

Documentation is created using github pages and mkdocs, [see](https://www.mkdocs.org/user-guide/deploying-your-docs/)

```bash
# to build locally
cd docs
pip install -r requirements.txt
mkdocs build
# to push to github pages
mkdocs gh-deploy
# if you want to run webserver locally
mkdocs serve
```