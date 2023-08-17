# Copyright (C) 2023 Antonio Rodriguez
# 
# This file is part of Transfer_Learning_for_CVD_risk_calculators.
# 
# Transfer_Learning_for_CVD_risk_calculators is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Transfer_Learning_for_CVD_risk_calculators is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Transfer_Learning_for_CVD_risk_calculators.
# If not, see <http://www.gnu.org/licenses/>.


import pandas as pd 
from typing import Tuple
import os   
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
import math 

def prepare_Steno(dataset_path : str = "", filename1 : str = "", filename2 : str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Read the Steno dataset from a .csv file and suit it to be processed 
    as a pd.DataFrame. This converted DataFrame is returned. 

    Args:
    -----
            dataset_path: path where dataset is stored. Set by default.
            filename1 : file name of the .csv containing the dataset. Set by default.
            fiename2 : file name of the .csv containing the regression Ground Truth 

    Returns:
    --------
            data: dataframe containing the whole dataset
            X : dataframe containing the dataset features
            Y : dataframe containing only the target variable
            cols_names: list of strings containing feature names. 
            y_tag: string containing target variable name.
    """

    # Go to dataset path
    os.chdir(dataset_path)

    # Open the .csv file and convert it into DataFrame
    data = pd.read_csv(filename1)
    gt = pd.read_csv(filename2)

    # Drop ID columns of both dataframes 
    data = data.drop(['id'], axis=1) 
    gt = gt.drop(['id'], axis=1) 

    # Concatenate features with GT 
    frames = [data, gt]
    data = pd.concat(frames, axis=1)
    
    # Only 5-year Cardiovascular risk is calculated, the other two columns are dropped
    data = data.drop(['cvd_risk_5y', 'eskd_risk_5y'], axis=1) 

    # Store column names 
    cols_names = data.columns

    # Store features' and target variable's names 
    cols_names_prev = data.columns
    y_tag = cols_names_prev[len(cols_names_prev)-1]
    cols_names = cols_names_prev[0:cols_names_prev.size]

    # Save X, Y, feature names and Y name 
    y_tag = cols_names[len(cols_names)-1]
    cols_names = cols_names[0:len(cols_names)-1]
    X = data[cols_names]
    Y = data[y_tag]
    
    return data, X, Y, cols_names, y_tag

def steno_original_risk_calculator(X : pd.DataFrame) -> pd.DataFrame: 

    """Computes a risk prediction of sufferin from Cardiovascular Diseases
    in 10 years as described by Vistisen et al., 2016 [1]. Formula can be 
    found in the Supplementary Material of the paper (available at
    https://www.ahajournals.org/action/downloadSupplement?doi=10.1161%2FCIRCULATIONAHA.115.018844&file=018844_supplemental_material.pdf).
    An interactive web-based risk engine can be found at  www.steno.dk/T1RiskEngine. 

    Args:
    -----
        X: features to compute the risk. 
       
    Returns:
    --------
        risk: dataframe the risk associate to each instance of X.

    References: 

    [1] Prediction of First Cardiovascular Disease Event in Type 1 Diabetes Mellitus, Circulation, 2016
    Vol. 133, No. 11, pp. 1-15, 2020. 
    """
    
    # Reset index to avoid future issues
    X.reset_index(drop=True, inplace=True)

    # Declaration of the dataframe to store the risk prediction of size X.shape[0]
    risk = pd.DataFrame(np.zeros(X.shape[0]))

    # Loop over the instances of X
    for i in range(X.shape[0]):
    #for i in range(1):

        # Albuminuria feature must be treated differently because it is a categorical feature
        # Variables used to compute the risk
        microalbuminuria = 0 
        macroalbuminuria = 0

        # Row of X to compute the risk
        row = X.iloc[i]
        
        if row['albuminuria'] == 0: 
            microalbuminuria = 0
            macroalbuminuria = 0
        if row['albuminuria'] == 1:
            microalbuminuria = 1
            macroalbuminuria = 0
        if row['albuminuria'] == 2: 
            microalbuminuria = 0
            macroalbuminuria = 1

        # Regression parameters 
        alpha = -5.716956267
        beta1 = 0.044302014
        beta2 = -0.208960873
        beta3 = 0.007769067 
        beta4  = 0.006514040
        beta5 = 0.117920173 
        beta6 = 0.008561294 
        beta7 = 0.295662570 
        beta8 = 0.532588474 
        beta9 = -0.504447543 
        beta10 = -0.443012444 
        beta11 = 0.308949210
        beta12 = 0.183553384 

        # Depending on the age (< 40 or >= 40), eGFR feature is treated diffferently.
        if row[('age')] < 40:
            egfr_factor = beta9*math.log(row['egfr'],2)
        else:
            egfr_factor = beta10*math.log(row['egfr'],2)
        
        r = np.exp(alpha + beta1*row['age'] +  beta2*row['sex'] + beta3*row['dm'] + beta4*row['sbp']
            + beta5*row['ldl'] + beta6*row['hba1c'] + beta7*microalbuminuria + beta8*macroalbuminuria
            + egfr_factor + beta11*row['smoking'] + beta12*row['excercise']) 
        
        risk.iloc[i] = 1 - np.exp(-r*10)

    return risk

def num2cat_Steno(data : pd.DataFrame):
    """This function replaces the numerical values corresponding to categories in 
    the Framingham database by its correspondant category. It returns a DataFrame
    after this replacement.

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    data['albuminuria']  = data['albuminuria'].replace([0,1,2],['albu_0','albu_1','albu_2'])
 
    return data 

def one_hot_enc_Steno(data):
    """This function performs One-Hot Encoding in the Steno database. 

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    
    # One-hot Encoder declaration 
    enc = OneHotEncoder(handle_unknown='ignore')
    # Albuminuria
    data[['albuminuria']] = data[['albuminuria']].astype('category')
    alb = pd.DataFrame(enc.fit_transform(data[['albuminuria']]).toarray())
    alb.columns = enc.categories_
    alb.reset_index(drop=True, inplace=True)

    # Drop target variable column to add it at the end 
    clas = data[['cvd_risk_5y']]
    clas.reset_index(drop=True, inplace=True)
    
    # Drop original categorical columns
    data = data.drop(['cvd_risk_5y'], axis=1)
    data.reset_index(drop=True, inplace=True)

    # Drop the original categorical column 
    data = data.drop(['albuminuria'], axis=1)
    data.reset_index(drop=True, inplace=True)

    # Joint one-hot encoding columns 
    data = data.join([alb, clas])
    
    return data
