import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

data = np.array(pd.read_csv('creditcard.csv'))
#print(data.shape)
y=data[:,-1];
#print(y.shape)
#print(y)
x = np.delete(data, -1, axis=1)
#print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

model= Sequential([Flatten(),Dense(16,activation='sigmoid',input_shape=(199364,30)),Dense(16,activation='sigmoid'),Dense(1,activation='sigmoid')])
model.compile(optimizer='sgd', loss='mean_squared_error',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=10, verbose = 1)
