import pickle
import numpy as np
import xgboost as xgb

from flask import Flask, request, jsonify

with open('xgb_model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('fraud')

def predict_model(dv, model, customer):
    columns = list(customer.keys())  # Access keys directly from the dictionary
    dicts = [customer]  # Wrap the customer dictionary in a list
    X = dv.transform(dicts)
    dtest = xgb.DMatrix(X, feature_names=columns)
    y_pred = model.predict(dtest)
    return y_pred

def predict_single(dv, model,customer):
    X = dv.transform([customer])
    y_pred = predict_model(dv, model,customer)
    return y_pred[0]

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single( dv, model, customer)
    fraud = prediction >= 0.5
    
    result = {
        'fraud_probability': float(prediction),
        'fraud': bool(fraud),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)