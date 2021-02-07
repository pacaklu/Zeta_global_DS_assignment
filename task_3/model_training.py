import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import gc
import lightgbm as lgb
import shap

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from math import sqrt


def comp_auc(target, predictions):
    return roc_auc_score(target, predictions)


def com_rsq(target, predictions):
    return sqrt(mean_squared_error(target, predictions))


def one_model(x_train, x_valid, y_train, y_valid, params):

    dtrain=lgb.Dataset(x_train, label=y_train)
    dvalid=lgb.Dataset(x_valid, label=y_valid)

    watchlist=dvalid

    booster=lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=watchlist,
        verbose_eval=200
    )

    return booster

def fit_model(x_train,y_train, x_valid, y_valid, params, preds): # fits Cross Validation

    models=[]
    performance = []
    '''
    folds=KFold(n_splits=CV_folds)
    models=[]
    performance = []

    for train_index, valid_index in folds.split(x_train):
        X_train, X_valid = x_train.loc[train_index,:], x_train.loc[valid_index,:] 
        Y_train, Y_valid = y_train.loc[train_index], y_train.loc[valid_index]

        X_train=X_train.reset_index(drop=True)
        X_valid=X_valid.reset_index(drop=True)

        Y_train=Y_train.reset_index(drop=True)
        Y_valid=Y_valid.reset_index(drop=True)


        print(X_train.shape)
        print(X_valid.shape)
        print(Y_train.shape)
        print(Y_valid.shape)
    '''


    model = one_model(x_train[preds], x_valid[preds], y_train, y_valid,params)
    models.append(model)

    x=[]
    x.append(model)
    predictions = predict(x, x_valid, preds)

    if (y_valid.nunique() == 2):     # classification
        performance.append(comp_auc(y_valid, predictions))
        
    else:
        performance.append(com_rsq(y_valid, predictions))


    print('Performance on validation sets:')
    print(performance)
    print('Mean:')
    print(np.mean(performance))

    return models

def predict(models, set_to_predict,preds):
    
    predictions=np.zeros(set_to_predict.shape[0])

    for model in models:
        predictions = predictions + model.predict(set_to_predict[preds])/len(models)

    return predictions

def comp_var_imp(models,preds):

    importance_df=pd.DataFrame()
    importance_df['Feature']=preds
    importance_df['Importance_gain']=0
    importance_df['Importance_weight']=0

    for model in models:
        importance_df['Importance_gain'] = importance_df['Importance_gain'] + model.feature_importance(importance_type = 'gain') / len(models)
        importance_df['Importance_weight'] = importance_df['Importance_weight'] + model.feature_importance(importance_type = 'split') / len(models)

    return importance_df

def plot_importance(models, imp_type, preds ,ret=False, show=True, n_predictors = 100):
    if ((imp_type!= 'Importance_gain' ) & (imp_type != 'Importance_weight')):
        raise ValueError('Only importance_gain or importance_gain is accepted')

    dataframe = comp_var_imp(models, preds)

    if (show == True):
        plt.figure(figsize = (20, len(preds)/2))
        sns.barplot(x=imp_type, y='Feature', data=dataframe.sort_values(by=imp_type, ascending= False).head(len(preds)))

    if (ret == True):
        return dataframe.sort_values(by=imp_type, ascending= False).head(len(preds))[['Feature', imp_type]]




def print_shap_values(preds, cols_num, cols_cat, x_train, y_train, x_valid, y_valid, params):
        
    x_train = x_train[preds]
    x_valid = x_valid[preds]

    
    for col in cols_cat:
        if x_train[col].isnull().sum()>0:
            x_train[col] = x_train[col].cat.add_categories('NA').fillna('NA')
            x_valid[col] = x_valid[col].cat.add_categories('NA').fillna('NA')
        _ , indexer = pd.factorize(x_train[col])
        x_train[col] = indexer.get_indexer(x_train[col])
        x_valid[col] = indexer.get_indexer(x_valid[col])
    

    model=one_model(x_train, x_valid, y_train, y_valid, params)
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_valid)

    if isinstance(shap_values, list):
               
        shap_values = shap_values[1]

    else:
        shap_values = shap_values


    shap.summary_plot(shap_values, x_valid)
    shap.summary_plot(shap_values, x_valid, plot_type='bar')
 
    var_imp_dataframe = {'Feature': preds, 'Shap_importance': np.mean(abs(shap_values),axis=0) }

    return x_valid, shap_values ,explainer
        



def shap_dependence_plot(set, cols_cat, shap_values, x, y=None):

    if  y is None:
        shap.dependence_plot(x, shap_values, set)

    else:            
        shap.dependence_plot(x, shap_values, set, interaction_index = y)

def prepare_roc():
    plt.figure(figsize=(10,10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
