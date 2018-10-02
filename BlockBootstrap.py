
# coding: utf-8

# In[33]:


import numpy as np
import math
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import TensorBoard
from time import time
import itertools
from math import factorial


# In[34]:


def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)


# In[35]:


def generateData(data, perm):
    n = data.shape[0]
    blockSize = n // len(perm)
    dataSet = np.zeros(data[0].shape)
    for i in perm:
        block = data[i*blockSize:(i+1)*blockSize]
#         print(block, end='\n\n')
        dataSet = np.vstack((dataSet, block))
    dataSet = dataSet[1:]
    return dataSet


# In[36]:


# dataframe = read_csv('sp500.csv')
fields = ['open', 'close']
dataframe = read_csv('GOOGL_data.csv', skipinitialspace = True, squeeze = True, usecols = fields)

# print(dataframe.head())
data = np.array(dataframe)
# print(data.shape)


# In[37]:


# dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
# print(data[:5])


# In[38]:


split = 0.75
trainSize = int(len(data)*split)
testSize = len(data)-trainSize
# print(trainSize)
# print(testSize)


# In[39]:


numInterval = 3
blockSize = trainSize//numInterval
# print(blockSize)


# In[40]:


train = data[0:trainSize,:]
test = data[trainSize:len(data),:]
# print(train[:5])


# In[41]:


# print(train, end="\n\n")
permutations = itertools.permutations(range(numInterval))
testBand = []
for perm in permutations:
    trainSet = generateData(train, perm)
    trainSet = np.reshape(trainSet, trainSet.shape)
    shuffleData = np.vstack((trainSet, test))

    lookBack = 3
    trainX, trainY = create_dataset(trainSet, lookBack)
    testX, testY = create_dataset(test, lookBack)

    # trainX = np.reshape(trainX, (trainX.shape[0], 2, trainX.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 2, testX.shape[1]))

    units = 100
    drop = 0.2
    epoch = 20

    model = Sequential()
    model.add(LSTM(units, input_shape=(lookBack, 2)))
    # model.add(Dropout(drop))
    model.add(Dense(2))
    model.compile(loss='mean_squared_error', optimizer='nadam')
#     model.summary()
#     tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
#     model.fit(trainX, trainY, epochs=epoch, batch_size=1, verbose=1, callbacks=[tensorboard])
    model.fit(trainX, trainY, epochs=epoch, batch_size=1, verbose=1)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)

    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    trainPredictPlot = np.empty_like(shuffleData)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookBack:len(trainPredict)+lookBack, :] = trainPredict

    testPredictPlot = np.empty_like(shuffleData)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(lookBack*2)+2:len(data)-1, :] = testPredict
    testBand.append(testPredict)
#     testPredictPlot[len(trainPredict)+(lookBack*2)+1:len(data)-1, :] = testPredict

    col = 0
    testPlot = np.empty_like(trainSet)
    testPlot[:, :] = 0
    testPlot = np.vstack((testPlot, test))
#     plt.plot(scaler.inverse_transform(shuffleData)[:,col])
#     plt.plot(trainPredictPlot[:,col], color = 'orange')
    plt.plot(testPredictPlot[:,col], color = 'green')
    plt.plot(scaler.inverse_transform(testPlot)[:,col], color = 'blue', linewidth = 0.3)
    plt.title('Epoch = %d Train = %.2f Test = %.2f' % 
            (epoch, trainScore, testScore))
# plt.show()


# In[42]:


temp = np.array(testBand)


# In[27]:


# temp.shape


# In[43]:


temp = np.reshape(temp, (temp.shape))
# print(temp)


# In[45]:


# temp[:,0,:]


# In[46]:


# print(np.mean(temp[:,0,:], axis = 0))
# print(np.std(temp[:,0,:], axis = 0))


# In[47]:


z_alpha = 1.96
n = factorial(numInterval)

confInterval = []

for i in range(temp.shape[1]):
    temp1 = temp[:, i, :]
    xBar = np.mean(temp1, axis = 0)
    s = np.std(temp1, axis = 0)
    l = xBar - 1.96*s/(n**0,.5)
    r = xBar + 1.96*s/(n**0,.5)
    pair = [l, r]
    confInterval.append(pair)


# In[49]:


# confInterval


# In[50]:


col = 0
offset = len(trainPredict)+(lookBack*2)+2
lower = []
upper = []
for i in range(len(confInterval)):
#     X = [i, i]
#     Y = [confInterval[i][0][0], confInterval[i][1][0]]
    lower.append(confInterval[i][0][0])
    upper.append(confInterval[i][1][0])
# XLower = np.array(range(offset+1, data.shape[0], blockSize//n))
# XUpper = np.array(range(offset+1, data.shape[0], blockSize//n))
XLower = np.array(range(offset+1, offset+1+temp.shape[1]))
XUpper = np.array(range(offset+1, offset+1+temp.shape[1]))
# print(XLower)
# print(np.array(upper)+offset)
# plt.plot(lower, XLower)
# plt.plot(upper, XUpper)
plt.plot(XLower, lower)
plt.plot(XUpper, upper)
plt.show()


# In[51]:


# np.array(range(offset+1, data.shape[0], blockSize//n))

