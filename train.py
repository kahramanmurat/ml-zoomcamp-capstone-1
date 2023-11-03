import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import pickle
from sklearn.model_selection import KFold

# Parameters
data_file = "data/creditcard_2023.csv"
output_file = "xgb_model.bin"
xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 2,
    'seed': 1,
    'verbosity': 1
}
n_splits = 5

def train_model(df_train, xgb_params):
    columns = list(df_train.columns[1:-1].values)
    dicts = df_train[columns].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    y_train = df_train["class"].values
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=columns)
    model = xgb.train(xgb_params, dtrain, num_boost_round=13)
    return dv, model

def predict_model(dv, model, df):
    columns = list(df.columns[1:-1].values)
    dicts = df[columns].to_dict(orient='records')
    X = dv.transform(dicts)
    dtest = xgb.DMatrix(X, feature_names=columns)
    y_pred = model.predict(dtest)
    return y_pred

def validate(df_full_train, n_splits, xgb_params):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []
    
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        dv, model = train_model(df_train, xgb_params)
        y_pred = predict_model(dv, model, df_val)
        auc = roc_auc_score(df_val["class"].values, y_pred)
        scores.append(auc)
    
    return scores

if __name__ == "__main__":
    df = pd.read_csv(data_file)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    
    print("Doing validation")
    validation_scores = validate(df_full_train, n_splits, xgb_params)
    print(f"Validation AUC scores: {validation_scores}")
    
    print("Training the final model")
    dv, model = train_model(df_full_train, xgb_params)
    y_test = df_test["class"].values
    y_pred = predict_model(dv, model, df_test)
    auc = roc_auc_score(y_test, y_pred)
    print(f"Final model AUC: {auc}")

    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)

    print(f"The model is saved to {output_file}")
