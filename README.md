# Beyond linear regression

_— Leveraging linear regression for feature selection of continuous/categorical variables—_


## Overview

This repository is a supplementary material for the medium article 
[**Beyond linear regression**: Leveraging linear regression for feature selection of continuous/categorical variables](https://towardsdatascience.com/beyond-linear-regression-467a7fc3bafb).

It applies the introduced feature selection technic on the [Automobile Data Set](https://archive.ics.uci.edu/ml/datasets/Automobile).
The objective is to find the top $K$ most relevant features that explains the price of the car.


## Project architecture

I adopt [cookiecutter Simple DS project](https://github.com/Badr-MOUFAD/cookiecutter-simple-DS-project) to structure this repository.

- ``data`` folder gathers raw and processed data
- ``notebooks`` contains the notebooks for preprocessing, exploring, and performing features selection.
- ``py_scripts`` is a python package where I put all the utils used in to produce the notebooks. 


## Get started

1. clone the repository on your local machine in ``cd`` to it

```shell
git clone https://github.com/Badr-MOUFAD/supp-material-med-article
cd supp-material-med-article
```

2. Initialize a the conda environnement and install ``py_scripts``

```shell
conda env create -f environment.yml
pip install -e .
```

3. run the notebooks in the ``notebooks`` folder in the specified order


## useful links

- Beyond linear regression medium article: https://towardsdatascience.com/beyond-linear-regression-467a7fc3bafb
- Automobile Dataset: https://archive.ics.uci.edu/ml/datasets/Automobile
- cookiecutter Simple DS project article: https://towardsdatascience.com/its-time-to-structure-your-data-science-project-1fa064fbe46
- ``celer`` documentation: https://mathurinm.github.io/celer/