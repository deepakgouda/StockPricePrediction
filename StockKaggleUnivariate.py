from pandas import read_csv, datetime, DataFrame, Series, concat
from math import sqrt
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

def parser(s):
	return datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f')

def shiftData(data, lag = 1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

def difference(data, interval = 1):
	diffSeries = list()
	for i in range(interval, len(data)):
		val = data[i] - data[i-interval]
		diffSeries.append(val)
	return Series(diffSeries)

def inverseDifference(history, yhat, interval = 1):
	return yhat + history[-interval]

def scale(train, test):
	scaler = MinMaxScaler(feature_range = (-1, 1))
	scaler = scaler.fit(train)
	train = train.reshape(train.shape[0], train.shape[1])
	trainScaled = scaler.transform(train)
	test = test.reshape(test.shape[0], test.shape[1])
	testScaled = scaler.transform(test)
	return scaler, trainScaled, testScaled
def invertScale(scaler, data, value):
	new_row = [x for x in data] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

def fitLSTM(train, batch_size = 1, epochs = 1000, neurons = 1):
	X = train[:, 0:-1]
	Y = train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, 
					X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss = "mean_squared_error", 
										optimizer = 'adam')
	for i in range(epochs):
		model.fit(X, Y, epochs = 1, batch_size = batch_size,
			verbose = 0, shuffle = False)
		model.reset_states()
	return model

def forecastLSTM(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size = batch_size)
	return yhat[0, 0]

fields = ['timestamp', 'open']
data = read_csv('log_inf_small.csv', header = 0, 
	parse_dates = [0], date_parser = parser, index_col = 0, 
	skipinitialspace = True, squeeze = True, usecols = fields)

rawValues = data.values
diffSeries = difference(rawValues, 1)

supervised = shiftData(diffSeries, 1)
supervisedValues = supervised.values

split = 60
train = supervisedValues[0:-split]
test = supervisedValues[-split:]

scaler, trainScaled, testScaled = scale(train, test)

batch = 1
epochs = 2000
neurons = 4
lstmModel = fitLSTM(trainScaled, batch, epochs, neurons)

trainReshaped = trainScaled[:, 0].reshape(
				len(trainScaled), 1, 1)
lstmModel.predict(trainReshaped, batch_size = batch)

predictions = list()
for i in range(len(testScaled)):
	X = testScaled[i, 0:-1]
	Y = testScaled[i, -1]
	yhat = forecastLSTM(lstmModel, batch, X)
	yhat = invertScale(scaler, X, yhat)
	yhat = inverseDifference(rawValues, yhat, len(testScaled)
						-i+1)
	predictions.append(yhat)
	expected = rawValues[len(train) + i]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, 
										yhat, expected))

rmse = sqrt(mse(rawValues[-split:], predictions))
print('Test RMSE: %.3f' % rmse)
plt.plot(rawValues[-split:])
plt.plot(predictions)
plt.show()