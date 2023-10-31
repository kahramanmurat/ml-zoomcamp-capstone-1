import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# parameters
random_state = 1
d = 5
eta = 0.1
n_estimators = 60 
scale_pos_weight = 1
output_file = 'xgb_model.bin'

# data preparation

df = pd.read_csv("data/creditcard_2023.csv")

df.columns = df.columns.str.lower().str.replace(" ", "_")

columns = list(df.dtypes.index)[1:-1]

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

X = df_full_train.copy()
y = X.pop("class")


# Validation
def cross_validation(model):
    
    #initiate prediction arrays and score lists
    train_scores, val_scores = [], []
    kf = StratifiedKFold(shuffle=True, random_state=random_state, n_splits = 5)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        #Train dataset
        X_train = X[columns].iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        #Validation dataset
        X_val = X[columns].iloc[val_idx]
        y_val = y.iloc[val_idx]
                
        #Train model    
        model.fit(X_train, y_train)
    
        #Predictions
        train_preds = model.predict_proba(X_train)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]
    
        #Evaluation for a fold
        train_score = roc_auc_score(y_train, train_preds)
        val_score = roc_auc_score(y_val, val_preds)
    
        #Saving the model score for a fold
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    print(f'val score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | train score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f}')
    return val_scores
 
model =  XGBClassifier(random_state = random_state,
                        n_estimators = n_estimators,
                        eta = eta, max_depth = d,
                        scale_pos_weight = scale_pos_weight)
score_list = cross_validation( model)

#final model 
XGBClassifier(random_state = random_state,
                        n_estimators = n_estimators,
                        eta = eta, max_depth = d,
                        scale_pos_weight = scale_pos_weight)
model.fit(X, y)
with open(output_file, 'wb') as f_out:
    pickle.dump((model), f_out)

print(f'the model is saved to {output_file}')