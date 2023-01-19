from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html") #will open home.html file from render_template folder


@app.route("/predict",methods=['POST','GET']) #after the input is being initialized it will redirect to predict page
def result():
    Age = int(request.form['Age']) #will request the value from the user dosent matter if its float value
    Sex = int(request.form['Sex']) #will request the value from the user
    ChestPainType = float(request.form['ChestPainType']) #will request the value from the user
    RestingBP = float(request.form['RestingBP']) #will request the value from the user
    Cholesterol = float(request.form['Cholesterol']) #will request the value from the user
    FastingBS = float(request.form['FastingBS']) #will request the value from the user
    RestingECG = float(request.form['RestingECG']) #will request the value from the user
    MaxHR = float(request.form['MaxHR']) #will request the value from the user
    ExerciseAngina = float(request.form['ExerciseAngina']) #will request the value from the user
    Oldpeak = float(request.form['Oldpeak']) #will request the value from the user
    ST_Slope = float(request.form['ST_Slope']) #will request the value from the user

    # reshaping all the inputs into 2d array and storing in x variable
    x = np.array([Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
        RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]).reshape(1,-1) 
    
    print("x: ", x)

    scaler_path= r'./models/sc.sav' #model path 

    sc = joblib.load(scaler_path) #load the scaling model

    X_std = sc.transform(x) #we do not need to fit because we already done fitting on home page

    model_path = r'./models/rf.sav' #random forest model path
    
    model = joblib.load(model_path) #load model

    Y_pred = model.predict(X_std) #where X_std is transformed version of x and predict the result
    print(Y_pred)
    print(type(Y_pred))
    if Y_pred == 0:
        response = "You Don't Have Heart Disease"
    else:
        response = "You Have Heart Disease"

    return jsonify({'Prediction': (response)})

if __name__ == "__main__":
    app.run(debug=True, port=9558)
