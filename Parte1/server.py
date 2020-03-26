# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:34:57 2020

@author: Alex Carneiro
"""

from flask import Flask, render_template, request

import numpy as np
import joblib

app = Flask(__name__)


model = joblib.load("models/model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    sepal_l = request.args.get('sepal_l')
    sepal_w = request.args.get('sepal_w')
    petal_l = request.args.get('petal_l')
    petal_w = request.args.get('petal_w')
    
    input_data = np.array([[float(sepal_l), float(sepal_w), float(petal_l), float(petal_w)]])
    result = model.predict(input_data)[0]
    
    return render_template('result.html',
                           sepal_l = sepal_l,
                           sepal_w = sepal_w,
                           petal_l = petal_l,
                           petal_w = petal_w, result = result)

if __name__ == '__main__':
    app.run()