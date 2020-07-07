from flask import render_template,Response,jsonify
from Melanoma import app
import pandas as pd
from flask import request
from Melanoma.Model import Model_One
import os
import numpy as np
import pdb

@app.route('/')
@app.route('/input')
def input():
   return render_template("input.html", title = 'Home')


demo_features = ['RIDAGEYR','NUMMAT', 'THICKNESS','tumor_location', 'RIAGENDR',  'ulceration']

def get_data():
    #pull 'data' from input field and store it
    global data

    # append all the values from input.HTML
    data_orginal = []
    for i in demo_features:
      tmp = request.args.get(i)
      data_orginal.append(tmp) 
   # split list of string by ',', return like ['56','1','2'..]
    data_str = []              
    for item in data_orginal:
      data_str.extend(item.split(","))  

   # convert string to float
    data = [float(i) for i in data_str]  
    data = np.array(data)  # convert to np.array
    return data


@app.route('/predict')
def predict():
    global risk
    global level
    data = get_data()
    risk = Model_One(data)
    if risk <= 0.35:
       level = 'Low'
    elif risk > 0.35 and risk < 0.5:
       level = 'Medium'
    else: 
       level = 'High'
    return render_template('Predict.html', the_result = risk, risk_level = level)

