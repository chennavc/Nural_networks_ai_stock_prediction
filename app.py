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
app = Flask(__name__)

@app.route("/")
def start():
    return render_template('index.html')

@app.route("/result",methods=['GET','POST'])
def result():
    if request.method == 'POST':
        symbol=request.form['symbol']
        msft = yf.Ticker(symbol)
        opk=msft.history(period="max")
        df=pd.DataFrame(opk)
        df1=opk['Close']
        plt.plot(opk.index,df1)
        plt.savefig('static/images/plot1.png')#ploting data
        plt.close() 

        #scaling data
        scaler=MinMaxScaler(feature_range=(0,1))
        df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
        
        #spliting data into training 
        training_size=int(len(df1)*0.65)
        test_size=len(df1)-training_size
        train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

        # convert an array of values into a dataset matrix
        def create_dataset(dataset, time_step=1):
	        dataX, dataY = [], []
	        for i in range(len(dataset)-time_step-1):
		        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		        dataX.append(a)
		        dataY.append(dataset[i + time_step, 0])
	        return numpy.array(dataX), numpy.array(dataY)
        
        # reshape into X=t,t+1,t+2,t+3 and Y=t+4
        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)

        # reshape input to be [samples, time steps, features] which is required for LSTM
        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

        ### Create the Stacked LSTM model
        model=Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
        model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam',metrics=["acc"])
        
        history = model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)
        #model learning

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('static/images/modelaccuracy.png')#ploting data
        plt.close() 

        #model loss

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('static/images/modelloss.png')#ploting data
        plt.close() 

        
        ### Lets Do the prediction and check performance metrics
        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)

        ### Calculate RMSE performance metrics
        math.sqrt(mean_squared_error(y_train,train_predict))

        ### Test Data RMSE
        math.sqrt(mean_squared_error(ytest,test_predict))

        # shift train predictions for plotting
        look_back=100
        trainPredictPlot = numpy.empty_like(df1)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        # shift test predictions for plotting
        testPredictPlot = numpy.empty_like(df1)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
        # plot baseline and predictions
        #plt.plot(scaler.inverse_transform(df1))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.savefig('static/images/plot2.png')#ploting data
        plt.close() 

        #prediction values
        x_input=test_data[(len(test_data)-100):].reshape(1,-1)
        x_input.shape
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()

        #next day preddiction
        

        lst_output=[]
        n_steps=100
        i=0
        while(i<30):
    
            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1
    

        #print(lst_output)

        day_new=np.arange(1,101)
        day_pred=np.arange(101,131)

        #ploting next 30 days preediction
        plt.plot(day_new,scaler.inverse_transform(df1[(len(df1)-100):]))
        plt.plot(day_pred,scaler.inverse_transform(lst_output))
        plt.savefig('static/images/plot3.png')#ploting data
        plt.close() 

        #30 days data
        opkdf=[]
        for x in range(1,30):
            NextDay_Date = datetime.datetime.today() + datetime.timedelta(days=x)
            opkdf.append(NextDay_Date)
        
        opkdf=pd.DataFrame(opkdf)
        opkdf["Date"]= pd.to_datetime(opkdf[0])
        ss=scaler.inverse_transform(lst_output)
        ss=pd.DataFrame(ss)
        opkdf["predicted close"]=ss
        opkdff=pd.DataFrame()
        opkdff["Date"]=opkdf["Date"]
        opkdff["predicted price"]=opkdf["predicted close"]

        return render_template ('result.html',  tables=[opkdff.to_html(classes='data')], titles=opkdff.columns.values)
    else:
        return render_template('index.html')
@app.route("/comingsoon",methods=['GET','POST'])
def comingsoon ():
    return render_template('comingsoon.html')



if __name__ == "__main__":
    app.run()