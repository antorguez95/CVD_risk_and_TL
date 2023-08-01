# Using Transfer Learning and Machine Learning to design a Cardiovascular Diseases risk calculator

## What's in this repository?

This repository contains the code of our work presented on the [26th Euromicro Conference Series on Digital System Design (DSD)](https://dsd-seaa2023.com/) in Durres, Albania, in September of 2023: 
[*"Novel Approach for AI-based Risk Calculator Development using Transfer Learning Suitable for Embedded Systems"*](https://dsd-seaa2023.com/). This works present a methodology for the preliminary
design of a risk calculator using tabular medical databases based on Machine Learning (ML), combining the knowledge of different clinically validated cardiovascular risk calculators using Transfer Learning, 
aiming a more personalized NCD risk estimation that the current regression-based methods. This works is enclosed in the [WARIFA European Project](https://www.warifa.eu/), whose main ojective is to 
develop an AI-based smartphone application to prevent chronic conditions, such as Diabetes Mellitus or Cardiovascular Diseases (CVD) by providing personalized recommendations depending on the subject and the variables that are collected from him. So, a preliminary basic high-level performance profiling has been also done to estimate the feasibility of implement this ML-based calculator in a micro-controller. 

The content of the scripts are described below: 

  - `Framingham_utils.py` and `Steno_utils.py`: data curation and preparation of the datasets.
  - `exploratory_data_analysis.py`: exploratory data analysis. 
  - `model_evaluation.py`: model evaluation functions for the selected ML models. 
  - `sdg_utils.py`: synthetic data generation functions used to balance the Framingham dataset
  - `train_utils.py`: functions to train the models. 
  - `profiling.py`: profiling functions extracted from [this example](https://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html#sphx-glr-auto-examples-applications-plot-prediction-latency-py)
  - `constants.py`: file with the name of the directories, file names, dataset names, and numerical and categorical features. 
  - `steno2fram.ipynb`and `fram2steno.ipynb`are the Python Notebooks that contain the framework itself. The former taking Steno database as reference, and the latter taking Framingham dataset. 

Please cite [our paper](https://dsd-seaa2023.com/) if this framework somehow helped you in your research and/or development work, or if you used this piece of code: 

*A. J. Rodriguez-Almeida et al., "Novel Approach for AI-based Risk Calculator Development using Transfer Learning Suitable for Embedded Systems" ___ doi: ____.*

## Datasets Availability

Both datasets are avilable under request to their authors (see [5] and [6] references in the [paper](https://dsd-seaa2023.com/) for Steno and Framingham availability, respectively).

## Set up before running this code

## How do I run these scripts?

## Generated results 

## Learn more
