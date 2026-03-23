from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

#Load model files
model = joblib.load("ctr_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    data = request.form.to_dict()

    #Extract timestamp properly
    timestamp = pd.to_datetime(data["Timestamp"])
    data["Hour"] = timestamp.hour
    data["DayOfWeek"] = timestamp.dayofweek
    data.pop("Timestamp")

    numeric_fields = ["Daily Time Spent on Site", 
                      "Age",
                      "Area Income",
                      "Daily Internet Usage",
                      "Hour",
                      "DayOfWeek"]
    
    for field in numeric_fields:
        data[field] = float(data[field])

    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns = columns, fill_value = 0)

    scaled_input = scaler.transform(input_df)

    probability = model.predict_proba(scaled_input)[0][1]
    prediction = model.predict(scaled_input)[0]

    result = "User Will Click" if prediction == 1 else "User Will Not Click"

    return render_template("index.html",
                           prediction_text = result,
                           probability_text = f"CTR Probability : {probability*100:.2f}",
                           developer_name = "Developed by Bhavana Pediredla")

if __name__ == "__main__":
    app.run(debug = True)
