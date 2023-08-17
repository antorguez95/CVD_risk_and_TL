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


import numpy as np 
from sklearn import metrics
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from sklearn import base

from typing import List, Tuple, Dict

import pandas as pd
 
def SVM_clas_eval(model : base.BaseEstimator, X : pd.DataFrame, y : pd.DataFrame):

      """Computes accuracy (acc), Area Under the Curve (AUC), Reciever Operating 
      Characteristics (ROC) and F1-score for a given SVM Classifier and a given 
      dataset. This function is thought to be used in a validation or test set".
  
      Args:
      -----
            models: sklearn SVM instance.
      Returns:
      --------
            acc: accuracy on the validation set.
            auc: AUC on the validation set.
            f1_sc: F1-score on the validation set.
      """
      # Prediction 
      y_pred = model.predict(X) 

      # Probabilities
      prob = model.predict_proba(X)
      
      # Accuracy 
      acc = metrics.accuracy_score(y, y_pred)
 
      # AUC
      y_score = model.decision_function(X)
      fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
      roc_auc = metrics.auc(fpr, tpr)
         
      # F1 Score
      f1_sc = f1_score(y, y_pred)

      # Recall 
      recall = metrics.recall_score(y, y_pred)

      # Precision
      prec =  metrics.precision_score(y, y_pred)
        
      return acc, roc_auc, f1_sc, prob, recall, prec

def MLP_clas_eval(model : base.BaseEstimator, X : pd.DataFrame, y : pd.DataFrame):

      """Computes accuracy (acc), Area Under the Curve (AUC), Reciever Operating 
      Characteristics (ROC) and F1-score for a given Multilayer Perceptron  Classifier and
      a given dataset. This function is thought to be used in a validation or test set".
  
      Args:
      -----
            models: sklearn SVM instance.
      Returns:
      --------
            acc: accuracy on the validation set.
            auc: AUC on the validation set.
            f1_sc: F1-score on the validation set.
      """
      # Prediction 
      y_pred = model.predict(X) 

      # Probabilities
      prob = model.predict_proba(X)
      
      # Accuracy 
      acc = metrics.accuracy_score(y, y_pred)
 
      # ROC
      disp = plot_roc_curve(model, X, y)
      
      # AUC
      auc = disp.roc_auc
         
      # F1 Score
      f1_sc = f1_score(y, y_pred)

      # Recall 
      recall = metrics.recall_score(y, y_pred)

      # Precision
      prec =  metrics.precision_score(y, y_pred)
        
      return acc, auc, f1_sc, prob, recall, prec

def perc_clas_eval(model : base.BaseEstimator, X : pd.DataFrame, y : pd.DataFrame):

      """Computes accuracy (acc), Area Under the Curve (AUC), Reciever Operating 
      Characteristics (ROC) and F1-score for a given Perceptron Classifier and
      a given dataset. This function is thought to be used in a validation or test set".
  
      Args:
      -----
            models: sklearn SVM instance.
      Returns:
      --------
            acc: accuracy on the validation set.
            auc: AUC on the validation set.
            f1_sc: F1-score on the validation set.
      """
      # Prediction 
      y_pred = model.predict(X) 

      # Accuracy 
      acc = metrics.accuracy_score(y, y_pred)
 
      # ROC
      disp = plot_roc_curve(model, X, y)
      
      # AUC
      auc = disp.roc_auc
         
      # F1 Score
      f1_sc = f1_score(y, y_pred)

      # Recall 
      recall = metrics.recall_score(y, y_pred)

      # Precision
      prec =  metrics.precision_score(y, y_pred)
        
      return acc, auc, f1_sc, recall, prec

def SVM_reg_eval(model : base.BaseEstimator, X : pd.DataFrame, y : pd.DataFrame):

      """Computes Maximum Absolute Error (MAE), Mean Square Error (MSE) and R2
      and F1-score for a given SVM Regressor and a given dataset. This 
      function is thought to be used in a validation or test set".
  
      Args:
      -----
            models: sklearn SVM instance.
      Returns:
      --------
            mae: MAE on the validation set.
            mse: MSE on the validation set.
            R2: R2 on the validation set.
      """
      # Prediction 
      y_pred = model.predict(X) 
      
      # MAE
      mae = metrics.mean_absolute_error(y, y_pred)

      # MSE
      mse = metrics.mean_squared_error(y, y_pred)

      # MSE
      r2 = metrics.r2_score(y, y_pred)
       
      return mae, mse, r2
		
def MLP_reg_eval(model : base.BaseEstimator, X : pd.DataFrame, y : pd.DataFrame):

      """Computes Maximum Absolute Error (MAE), Mean Square Error (MSE) and R2
      and F1-score for a given Perceptron Classifier and a given dataset. This 
      function is thought to be used in a validation or test set".
  
      Args:
      -----
            models: sklearn SVM instance.
      Returns:
      --------
            mae: MAE on the validation set.
            mse: MSE on the validation set.
            R2: R2 on the validation set.
      """
      # Prediction 
      y_pred = model.predict(X) 
      
      # MAE
      mae = metrics.mean_absolute_error(y, y_pred)

      # MSE
      mse = metrics.mean_squared_error(y, y_pred)

      # MSE
      r2 = metrics.r2_score(y, y_pred)
       
      return mae, mse, r2

def acc_auc_roc_general(model : base.BaseEstimator, X : pd.DataFrame, y : pd.DataFrame):

      """Computes accuracy (acc), Area Under the Curve (AUC), Reciever Operating 
      Characteristics (ROC) and F1-score for a sklearn Classifier and a given 
      dataset. This function is thought to be used in a validation or test set".
  
      Args:
      -----
            models: sklearn classifier instance.
            X : features of the dataset
            y : target variable of the dataset. 
      Returns:
      --------
            acc: accuracy on the validation set.
            auc: AUC on the validation set.
            f1_sc: F1-score on the validation set.
      """
      # Prediction 
      y_pred = model.predict(X) 
      
      # Accuracy
      acc = metrics.accuracy_score(y, y_pred)
                       
      # ROC
      disp = plot_roc_curve(model, X, y)
      
      # AUC
      auc = disp.roc_auc

      # F1 Score
      f1_sc = f1_score(y, y_pred)
              
      return acc, auc, f1_sc 

def conf_matrix(models):
      for model, data in models:
        y_pred = model[0].predict(model[1])
        confusion_matrix(data[0], y_pred)
        title = "Confusion Matrix - %s" % data[1]
        disp = plot_confusion_matrix(model[0], model[1], data[0],
                                  cmap=plt.cm.Blues)
        disp.ax_.set_title(title)
        plt.show()
        conf_matrix = disp.confusion_matrix
      return conf_matrix
     
def get_eval_dictionaries(datasets: List, classifiers : Dict, regressors : Dict,
                                            class_metrics : List, reg_metrics : List) -> Tuple [Dict, Dict, Dict]:
      """This function generates and returns one dictionary to evaluate the
      classification performance employing and its associated hyperparameters
      four different classic Machine Learning (ML) models: SVM, RF, XGB and
      KNN. There are two datasets that approach a classification task and 
      another one that approaches a regression task. F1-score, ROC and AUC has
      been used for the former and MAE for the later. 


      Args:
      -----
            datasets: list of the datasets used to evaluate these classifiers.
            classifiers: empty dictionary containing the abovementioned ML classifiers. 
            regressors: empty dictionary the abovementioned ML regressors (only for Steno dataset)
            class_metrics: classification metrics to be computed 
            reg_metrics: regression metrics to be computed  

      Returns:
      --------
            metrics: dictionary of structure {dataset: 
                                                      {ML classifier/regressor : 
                                                            {Classification/Regression metric: Dict}
                                                            {Best parameters:  Dict}}}
                                    
      """           
                  
      metrics = {}  
      hp = {'best_hyperparameters':{}}        
      
      for data in datasets : 
      
            # Steno is a regression approach, the rest are classification ones 
            if 'Steno' in data: 
                  metrics[data] = regressors

                  for regressor in regressors:  
                        metrics[data][regressor] = reg_metrics, hp
            
            else: 
                  metrics[data] = classifiers

                  for classifier in classifiers:  
                        metrics[data][classifier] = class_metrics, hp

      return metrics