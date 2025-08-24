from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("income_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    age = data["age"]
    education = data["education"]
    hours = data["hours"]

    features = np.array([[age, education, hours]])
    prediction = model.predict(features)[0]

    result = "Income > 50K" if prediction == 1 else "Income <= 50K"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
