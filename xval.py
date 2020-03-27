#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score


# In[2]:


# load the data to pandas frame
data = pd.read_csv('./data/creditcard.csv')
data


# In[3]:


data.drop('Time', axis=1, inplace=True)
data


# In[4]:


def binary_class_average_precision_score(y_pred, data):
    y_true = data.get_label()
    return 'average-precision', average_precision_score(y_true, y_pred), True


# In[5]:


y = data['Class']
x = data.drop('Class', axis=1)

print('x.shape', x.shape)
print('y.shape', y.shape)


# In[6]:


def objective(trial):
    param = {
        'objective':'binary',
        'verbose':-1,
        'metric': 'average-precision',
        #'device': 'gpu',
        #'is_unbalance': True,
        'num_leaves': trial.suggest_int('num_leaves', 100, 700),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 300),
        'max_bin': trial.suggest_int('max_bin', 200, 4000),
        
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.01, 1.0),
        'lambda_l1': trial.suggest_uniform('lambda_l1', 0.0, 80.0),
        'lambda_l2': trial.suggest_uniform('lambda_l2', 0.0, 80.0),
        'min_gain_to_split': trial.suggest_uniform('min_gain_to_split', 0.0, 1.0),
        'scale_pos_weight': trial.suggest_uniform('scale_pos_weight', 1.0, 10.0), 
        }
    
    if trial.suggest_categorical('do_bagging', [True, False]):
        param['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.01, 0.99)
        param['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 40)    
    
    print('param', param)
    
    boosting_losses = []    
    first_xval_step = True
    
    xval = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in xval.split(x, y):
        x_train = x.iloc[train_index]
        y_train = y.iloc[train_index]
        x_val = x.iloc[test_index]
        y_val = y.iloc[test_index]
        
        train_data = lgb.Dataset(x_train, label=y_train)
        val_data = lgb.Dataset(x_val, label=y_val) 
        
        evals_result = {}
        
        if first_xval_step:
            num_boost_round = 1000
            early_stopping_rounds = 35
            first_xval_step = False
        else:
            num_boost_round = len(boosting_loss)
            early_stopping_rounds = None

        bst = lgb.train(
            param, 
            train_data, 
            valid_sets=[val_data], 
            verbose_eval=False,
            evals_result=evals_result,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            keep_training_booster = True,
            feval=binary_class_average_precision_score,
            )        
        
        boosting_loss = np.asarray(evals_result['valid_0']['average-precision'])
        #print(boosting_loss)
        boosting_losses.append(boosting_loss)

    boosting_losses = np.stack(boosting_losses)
    #print('boosting_losses', boosting_losses)
    mean_losses = np.mean(boosting_losses, axis=0)
    #print('mean_losses', mean_losses)
    max_mean_loss = max(mean_losses)
    best_avg_boosting_round = np.argmax(mean_losses)
    print('best_avg_boosting_round', best_avg_boosting_round)
    print('num_boost_round', num_boost_round)
    print('max_mean_loss', max_mean_loss)
    
    trial.set_user_attr('mean_losses', mean_losses.tolist())
    trial.set_user_attr('best_avg_boosting_round', int(best_avg_boosting_round))

    return max_mean_loss
    
                


# In[8]:


study = optuna.create_study(
        direction='maximize', 
        study_name='test_01', 
        storage='sqlite:///data/training_03.db',
        load_if_exists=True,
        )


# In[9]:


# Hyperparameter Optimierung starten
study.optimize(objective)


# In[ ]:




