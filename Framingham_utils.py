# Copyright (C) 2023 Antonio Rodriguez
# 
# This file is part of CVD_risk_and_TL.
# 
# CVD_risk_and_TL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# CVD_risk_and_TL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CVD_risk_and_TL.
# If not, see <http://www.gnu.org/licenses/>.

# Dependencies 
import os 
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder

def prepare_Framingham(dataset_path : str, filename : str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Read the Framingham dataset from a .csv file and suit it to be processed 
    as a pd.DataFrame. This converted DataFrame is returned. 

    Args:
    -----
            dataset_path: path where dataset is stored.
            filename : file name of the .csv containing the dataset.

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
    data = pd.read_csv(filename)

    # Store column names 
    cols_names = data.columns
    
    # Replace nan values by np.nan and 
    data.replace(('nan'), (np.nan), inplace=True)

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

def numerical_conversion_Framingham(data : np.array, features : str, y_col : str):
    """Fix all Framingham database features data types to its original type after KNNImputer is used,
    since this functions returns only a floating points ndarray. For more, check sklearn 
    documentation of this function at
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html. After 
    fixing datatypes, an ndarray to pd.DataFrame conversion is performed. Notice that this
    operation is only done in the fields that were not originally floats.

    Args:
    -----
            data: data returned by KNN Imputer (float data type).
            features: list of strings containing the feature names of the dataset. 
            y_col: target variable (i.e., Y) name 

    Returns:
    --------
            data: dataframe containing the whole dataset after imputation
            X : dataframe containing the dataset features after imputation
            y : dataframe containing only the target variable after imputation
    """
    # From ndarray to pd.DataFrame
    names = features.insert(len(features), y_col)
    data = pd.DataFrame(data, columns = names)
    
    # Fixing necessary datatypes to int (including categorical variables)
    data['male'] = data['male'].astype(int)
    data['age'] = data['age'].astype(int)
    data['education'] = data['education'].astype(int)
    data['currentSmoker'] = data['currentSmoker'].astype(int)
    data['cigsPerDay'] = data['cigsPerDay'].astype(int)
    data['BPMeds'] = data['BPMeds'].astype(int)
    data['prevalentStroke'] = data['prevalentStroke'].astype(int)
    data['prevalentHyp'] = data['prevalentHyp'].astype(int)
    data['diabetes'] = data['diabetes'].astype(int)
    data['totChol'] = data['totChol'].astype(int)
    data['sysBP'] = data['sysBP'].astype(float)
    data['diaBP'] = data['diaBP'].astype(float)
    data['BMI'] = data['BMI'].astype(float)
    data['heartRate'] = data['heartRate'].astype(int)
    data['glucose'] = data['glucose'].astype(int)
    data['TenYearCHD'] = data['TenYearCHD'].astype(int)
    
    # Separate X and Y 
    X = data[features]
    y = data[[y_col]]    
     
    return data, X, y

def general_conversion_Framingham (data : pd.DataFrame) -> pd.DataFrame :
    """Fix all Framingham database features data types to its original type.
    Categorical variables are set as "object" type. Binary ones as "bool".
    A DataFrame with the original datatypes of this database is returned.

    Args:
    -----
            data: dataset with datatypes not corresponding to the original ones.

    Returns:
    --------
            data: dataframe with the original datatypes 
    """
    data['male'] = data['male'].astype(int)
    data['age'] = data['age'].astype(int)
    data['education'] = data['education'].astype('object')
    data['currentSmoker'] = data['currentSmoker'].astype(int)
    data['cigsPerDay'] = data['cigsPerDay'].astype(int)
    data['BPMeds'] = data['BPMeds'].astype(int)
    data['prevalentStroke'] = data['prevalentStroke'].astype(int)
    data['prevalentHyp'] = data['prevalentHyp'].astype(int)
    data['diabetes'] = data['diabetes'].astype(int)
    data['totChol'] = data['totChol'].astype(int)
    data['sysBP'] = data['sysBP'].astype(float)
    data['diaBP'] = data['diaBP'].astype(float)
    data['BMI'] = data['BMI'].astype(float)
    data['heartRate'] = data['heartRate'].astype(int)
    data['glucose'] = data['glucose'].astype(int)
    data['TenYearCHD'] = data['TenYearCHD'].astype(int)
    
    return data

def num2cat_Framingham(data : pd.DataFrame):
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
    data['education']  = data['education'].replace([1,2,3,4],['edu1','edu2','edu3','edu4'])
 
    return data 

def one_hot_enc_Framingham(data):
    """This function performs One-Hot Encoding in the Framingham database. 

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe after One-Hot Encoding  
    """
    
    # One-hot Encoder declaration 
    enc = OneHotEncoder(handle_unknown='ignore')
    # education
    data[['education']] = data[['education']].astype('category')
    edu = pd.DataFrame(enc.fit_transform(data[['education']]).toarray())
    edu.columns = enc.categories_
    edu.reset_index(drop=True, inplace=True)

    # Drop target variable column to add it at the end 
    clas = data[['TenYearCHD']]
    clas.reset_index(drop=True, inplace=True)
    
    # Drop original categorical columns
    data = data.drop(['TenYearCHD'], axis=1)
    data.reset_index(drop=True, inplace=True)

    # Drop the original categorical column 
    data = data.drop(['education'], axis=1)
    data.reset_index(drop=True, inplace=True)

    # Joint one-hot encoding columns 
    data = data.join([edu, clas])
    
    return data