# Using Transfer Learning and Machine Learning to design a Cardiovascular Diseases risk calculator

## What's in this repository?

This repository contains the code of our work presented on the [26th Euromicro Conference Series on Digital System Design (DSD)](https://dsd-seaa2023.com/) in Durres, Albania, in September of 2023: 
[*"Novel Approach for AI-based Risk Calculator Development using Transfer Learning Suitable for Embedded Systems"*](https://dsd-seaa2023.com/). This works presents a methodology for the preliminary
design of a risk calculator using medical tabular databases based on Machine Learning (ML), combining the knowledge of different clinically validated cardiovascular risk calculators using Transfer Learning. This aims a more personalized NCD risk estimation than the current regression-based approaches. This work is enclosed in the [WARIFA European Project](https://www.warifa.eu/), whose main ojective is to 
develop an AI-based application to prevent chronic conditions, such as Diabetes Mellitus or Cardiovascular Diseases (CVD), by providing personalized recommendations depending on the subject and the variables that are collected from him. So, a preliminary basic high-level performance profiling has been also done to estimate the feasibility of implementing this ML-based calculator in a micro-controller. 

The content of the scripts are described below: 

  - `Framingham_utils.py` and `Steno_utils.py`: data curation and preparation of the datasets.
  - `exploratory_data_analysis.py`: exploratory data analysis. 
  - `model_evaluation.py`: model evaluation functions for the selected ML models. 
  - `train_utils.py`: functions to train the models. 
  - `profiling.py`: profiling functions extracted from [this example](https://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html#sphx-glr-auto-examples-applications-plot-prediction-latency-py)
  - `constants.py`: file with the name of the directories, file names, dataset names, and numerical and categorical features. MUST BE CHANGED WITH YOUR OWN PATHS, FILES, ETC!!!
  - `steno2fram.ipynb`and `fram2steno.ipynb`are the Python Notebooks that contain the framework itself. The former taking Steno database as reference, and the latter taking Framingham dataset. 

Please cite [our paper](https://dsd-seaa2023.com/) if this framework somehow helped you in your research and/or development work, or if you used this piece of code: 

*A. J. Rodriguez-Almeida et al., "Novel Approach for AI-based Risk Calculator Development using Transfer Learning Suitable for Embedded Systems" ___ doi: ____.*

## Datasets Availability

Both datasets are avilable under request to their authors (see [5] and [6] references in the [paper](https://dsd-seaa2023.com/) to check Steno and Framingham availability, respectively).

## Requirements to run this code

This code was developed with Python ___, with `ipykernel` installed to run the framework using Jupyter Notebooks, so this feature must be supported by your software development tool. 

## How do I run these scripts?

After changing the paths, filenames, etc. from `constants.py` to the corresponding ones of your paths, you just have to run one of the `ipynb` files in the development environment you use. 

## Generated results 

The execution of each `.ipynb`file generates the `EDA` and `results` folders. Mainly, in the `EDA` folder, the histograms of the different continous variables of the datasets are stored, to visually demonstrate the heterogeneity of both datasets. In the `results` folder, an Excel file contiaining the results pre-TL and post-TL are placed, including also the classification confusion matrices. Please, refer to our [paper](https://dsd-seaa2023.com/) for a more detailed analysis of the obtained results. 

## Learn more

For any other questions related with the code or the [proposed framework](https://dsd-seaa2023.com/) itself, you can post an issue on this repository or contact me via email.
