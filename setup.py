#!/usr/bin/env python3
# flake8: noqa
import os

import setuptools


def requirements(fname):
    return [
        line.strip() for line in open(os.path.join(os.path.dirname(__file__), fname))
    ]


pwd = os.path.dirname(__file__)


with open(os.path.join(pwd, "VERSION")) as f:
    version = f.read().strip()
    assert len(version.split(".")) == 3, "bad version spec"
    majorminor = version.rsplit(".", 1)[0]

extras = {}
extras["test"] = ["pytest", "pytest-cov", "codecov"]
extras["docs"] = requirements("docs/requirements.txt")
extras["quality"] = [
    "black",
    "isort",
    "flake8",
]
extras["all"] = extras["test"] + extras["docs"] + extras["quality"]
reqs = requirements("requirements.txt")

# main setup kw args
setup_kwargs = {
    "name": "churn_pred",
    "version": version,
    "description": "Churn predictor",
    "long_description": open("README.MD", "r", encoding="utf-8").read(),
    "long_description_content_type": "text/markdown",
    # "long_description": long_description,
    "author": "Pavol Mulinka",
    "author_email": "mulinka.pavol@gmail.com",
    "url": "https://github.com/5uperpalo/churnpred",
    "license": "MIT",
    "install_requires": reqs,
    "extras_require": extras,
    "python_requires": ">=3.8.1",
    "classifiers": [
        "Environment :: Other Environment",
        "Framework :: Jupyter",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    "zip_safe": True,
    "package_data": {"churnpred": ["data/*"]},
    "packages": setuptools.find_packages(exclude=["test_*.py"]),
}


if __name__ == "__main__":
    setuptools.setup(**setup_kwargs)
