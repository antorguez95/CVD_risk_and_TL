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
from sklearn import base
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Dict, List, Tuple

def train_test_split_and_concat(X : pd.DataFrame, y: pd.DataFrame, test_size : float = 0.2, random_state : int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a dataset into training and test sets. Returns the training and test sets already 
    concatenated. 
    
    Args:
    -----
       X: The features/independent variables of a given dataset.
       y: The target/dependent variable of a given dataset
       test_size: proportion of the dataset to be used as test set. Defaults to 0.2. 
       random_state: random state used to shuffle the dataset. Defaults to 4.
    
    Returns:
    --------
       X_train: The features/independent variables of the training set.
       y_train: The target/independent variable of the training set.
       X_test: The features/independent variables of the test set.
       y_test: The target/independent variable of the test set.
    """
    
    # Call to the split function of the sklearn library
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
    # Concatenes the training and test sets
    train_data = pd.concat([X_train, y_train], axis = 1)
    validation_data = pd.concat([X_test, y_test], axis = 1)
    
    return X_train, X_test, y_train, y_test, train_data, validation_data

def train_classifier(model : base.BaseEstimator, params : Dict, model_name: str, 
              dataset : str, X : pd.DataFrame , Y : pd.DataFrame,  cv : int = 10,
              scoring : str = 'f1') -> Tuple [base.BaseEstimator, List, pd.DataFrame, Dict]:
       """Trains a model using Grid Search hyperparameters optimization strategy.
    
       Args:
       -----
              model: sklearn estimator instance.
              params: Hyperparameters to be tuned (model dependent).
              model_name: ML model name. 
              dataset : name of the dataset used to train the ML model 
              X: The features/independent variables of a given dataset.
              Y: The target/dependent variable of a given dataset
              params: Hyperparameters to be tuned (model dependent).
              cv: number of folds in the cross-validation. Defaults to 10. 
              scoring : target metric to evaluate the performance of croass-validated model. Defaults to 'f1'.


       Returns:
       --------
              best_model: estimator that shows best cross-validation performance.
              results: summary of best estimator results. 
              overall_results: summary of overall results.
              best_params: best hyperparameters combination.
       """
       
       # Grid Search hyperparameters optimization 
       grid = GridSearchCV(model,
                        params,
                        scoring = scoring,
                        cv = cv,
                        n_jobs = -1,
                        return_train_score = True,
                        verbose = 0).fit(X,Y)

       a = grid.cv_results_
       b = pd.DataFrame.from_dict(a, orient='columns')
       
       # Dropping worthless columns 
       c = b.drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 
                   'std_score_time','split0_test_score',
                   'split1_test_score', 'split2_test_score', 'split3_test_score',
                   'split4_test_score', 'split5_test_score', 'split6_test_score',
                   'split7_test_score', 'split8_test_score', 'split9_test_score',
                   'split0_train_score', 'split1_train_score', 'split2_train_score',
                   'split3_train_score', 'split4_train_score', 'split5_train_score',
                   'split6_train_score', 'split7_train_score', 'split8_train_score',
                   'split9_train_score', 'mean_train_score', 'std_train_score'], axis = 1)
       
       # Save results from best test score to worst
       overall_results = c.sort_values('rank_test_score')

       # Save best model 
       best_model = grid.best_estimator_

       # Save the best parameters 
       best_params = grid.best_params_

       # Cross validation score of the best model 
       results = [cross_val_score(best_model, X, Y, cv = cv, scoring=scoring).mean(), cross_val_score(model, X, Y, cv = cv, scoring=scoring).std()] 
       
       # Printing training sets results 
       print(model_name ,": ", dataset," -> ", scoring, "on training set is", results[0],"(",results[1],")")
       
       return best_model, results, overall_results, best_params 

def standardization(X: pd.DataFrame, y: pd.DataFrame, numerical_variables : List) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
       """Performs standardization in the a dataset. It fits a dataset computing 
       its mean and std., and use them to fit the target dataset. This function only 
       works with numerical features.

       Args:
       -----
              X: The features to be standardized.
              y: target variable. 
              numerical_variables : list containing the numerical features of the dataset to be standardized

       Returns:
       --------
              data: standardized dataset.
              X: standardized features .
              y: target variable. 
       """
    
       # Declare columns to further use 
       columns = X.columns

       X[numerical_variables] = StandardScaler().fit(X[numerical_variables]).transform(X[numerical_variables].astype(float))
       X_norm = X

       # To DataFrame to concatenate with Y
       X_norm = pd.DataFrame(X_norm, columns = columns)

       # Concatenate X and Y
       data = pd.concat([X_norm, y], axis = 1)

       return data, X_norm, y

def cross_standardization(X_src : pd.DataFrame, X_trg : pd.DataFrame, y: pd.DataFrame,
       numerical_variables : List) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
       """Performs cross standardization in the a dataset. It fits a dataset computing 
       its mean and std., and use them to fit the target dataset. This function only 
       works with numerical features.

       Args:
       -----
              X_src: dataset used to fit the scaler
              X_trg: dataset to be standardized.
              y: target variable. 
              numerical_variables : list containing the numerical features of the dataset to be standardized

       Returns:
       --------
              data: standardized dataset.
              X: standardized features .
              y: target variable. 
       """
    
       # Declare columns to further use 
       columns = X_src.columns
       
       # Standardization of the target dataset

       # Standardizaton of the target dataset 
       X_trg[numerical_variables] = StandardScaler().fit(X_src[numerical_variables]).transform(X_trg[numerical_variables].astype(float))
       X_norm = X_trg

       # To DataFrame to concatenate with Y
       X_norm = pd.DataFrame(X_norm, columns = columns)

       # Concatenate X and Y
       data = pd.concat([X_norm, y], axis = 1)

       # y = y.to_numpy()
       # y = y.ravel() # To avoid warning. We go from a column vector to a 1D-array
    
       return data, X_norm, y


def one_hot_enc(data : pd.DataFrame, y_tag : str, categorical_features : List) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
       """This function performs One-Hot Encoding to the indicated "categorical_features". 

       Args:
       -----
              data: dataset.
              y_tag : name of the target variable
              categorical_fueatures: list with the features to be performed one-hot-encoding

       Returns:
       --------
              data: dataframe with the categories represented by their correspondant string  
       """
    
       # One-hot Encoder declaration 
       enc = OneHotEncoder(handle_unknown='ignore')

       # Drop target variable column to add it at the end 
       clas = data[[y_tag]]
       clas.reset_index(drop=True, inplace=True)
       data = data.drop(clas, axis=1)
       data.reset_index(drop=True, inplace=True)
              
       for feat in categorical_features : 
              
              data[[feat]] = data[[feat]].astype('category')
              cat = pd.DataFrame(enc.fit_transform(data[[feat]]).toarray())
              cat.columns = enc.categories_
              cat.reset_index(drop=True, inplace=True)

              # Drop original column 
              data = data.drop([feat], axis=1)
              data.reset_index(drop=True, inplace=True)

              # Joint one-hot encoding columns 
              data = data.join([cat])
              data.reset_index(drop=True, inplace=True)

       X = data
       
       # Joint the last column 
       data = data.join([clas])
        
       return data, X, clas
