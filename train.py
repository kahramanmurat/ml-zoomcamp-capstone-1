#!/usr/bin/env python
# coding: utf-8

## data location
## "https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023"

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# parameters

C = 0.1
n_splits = 5
output_file = f"model_C={C}.bin"

# data preparation

df = pd.read_csv("data/creditcard_2023.csv")

df.columns = df.columns.str.lower().str.replace(" ", "_")

columns = list(df.dtypes.index)[1:-1]

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# training


def train(df_train, y_train, C=C):
    dicts = df_train.to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df.to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# validation

print(f"doing validation with C={C}")

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train["class"].values
    y_val = df_val["class"].values

    dv, model = train(df_train[columns], y_train, C=C)
    y_pred = predict(df_val[columns], dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f"auc on fold {fold} is {auc}")
    fold = fold + 1


print("validation results:")
print("C=%s %.3f +- %.3f" % (C, np.mean(scores), np.std(scores)))


# training the final model

print("training the final model")

dv, model = train(df_full_train[columns], df_full_train["class"].values, C=0.1)
y_pred = predict(df_test[columns], dv, model)

y_test = df_test["class"].values
auc = roc_auc_score(y_test, y_pred)

print(f"auc={auc}")


# Save the model

with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f"the model is saved to {output_file}")
