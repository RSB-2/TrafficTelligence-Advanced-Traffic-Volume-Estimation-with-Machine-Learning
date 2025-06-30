import numpy as np
import pandas as pd
import pickle
import os
from flask import Flask, request, render_template

app = Flask(__name__)


model_path = r"C:\Users\ASUS\OneDrive\Desktop\TRAFFIC\model.pkl"
scaler_path = r"C:\Users\ASUS\OneDrive\Desktop\TRAFFIC\scaler.pkl"

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        holiday = int(request.form['holiday'])
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = int(request.form['weather'])
        day = int(request.form['day'])
        month = int(request.form['month'])
        year = int(request.form['year'])

       
        features = np.array([[holiday, temp, rain, snow, weather, day, month, year]])

     
        scaled_features = scaler.transform(features)

    
        prediction = model.predict(scaled_features)[0]
        output = int(prediction)

        return render_template('index.html', prediction_text=f"Estimated Traffic Volume: {output}")

    except Exception as e:
        return render_template('index.html', prediction_text=f" Error occurred: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
