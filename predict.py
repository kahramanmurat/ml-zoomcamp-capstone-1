import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = "model_C=1.0.bin"

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("fraud")


@app.route("/predict", methods=["POST"])
def predict():
    credit_card = request.get_json()

    X = dv.transform([credit_card])
    y_pred = model.predict_proba(X)[0, 1]
    credit_card = y_pred >= 0.5

    result = {"fraud_probability": float(y_pred), "fraud": bool(credit_card)}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
