<h1 align="center">Surface Topography from Interferences</h1>
<p align="center">
  

<img src="https://img.shields.io/badge/made%20by-PUT-blue.svg" >

<img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" >

<img src="https://img.shields.io/github/stars/Hatter01/topography-scratch.svg?style=flat">

<img src="https://img.shields.io/github/issues/Hatter01/topography-scratch.svg">
</p>

## Description
The mission of our project is to improve the inferometer optimization system. Based on the generated data and preliminary code created by the group  of [Krzysiek](https://github.com/krzysztofmarchewkaa) and [Paweł](https://github.com/nickerror) in their [repository](https://github.com/nickerror/InterferometrRepo), we hope to estimate the distance of the inferometer from the object with as much precision as possible.

## Dataset
 We will be using CNNs on thousands of images showing the distribution of wave amplitudes such as this one. 
<p align="center">
<img src= "./src/readme_assets/wygenerowany_zaszumiony.png" width="50%">

## About us
We are continuing this project as part of passing the Problem Class 1 course taught by Dabrze. We ourselves are 3rd. year Artificial Intelligence students at Poznań University of Technology.

## Project setup

```

```

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
