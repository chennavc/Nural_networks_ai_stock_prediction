from flask import Flask,render_template,request
# from flask_jsonpify import jsonpify
import yfinance as yf
import pandas as pd
from pandas import DataFrame, read_csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
import numpy
import numpy as np
from numpy import array
import keras
import datetime
import ml
app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route("/",methods=['GET','POST'])
def home():
    if request.method == 'POST':
        symbol=request.form['symbol']
        opkdff=ml.prediction(symbol)

        return render_template ('result.html',  tables=[opkdff.to_html(classes='data')], titles=opkdff.columns.values)
    else:
        return render_template('index.html')
@app.route("/comingsoon",methods=['GET','POST'])
def comingsoon ():
    return render_template('comingsoon.html')



if __name__ == "__main__":
    app.run(debug=True)