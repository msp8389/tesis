# importamos las librerias generales 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
# librerias para deep learning
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight


# carga de datos
datos = pd.read_csv('dataset.csv')
predict = pd.read_csv('predict2.csv')

entrenamiento = datos.values
print(entrenamiento[0])
# Elimina todos las columnas en las posiciones 0 y 24
x_entrenamiento = np.delete(entrenamiento, [0, 24], 1)
print(x_entrenamiento[0])

# X= dataset.iloc[:,0:8]
# y= dataset.iloc[:,8]

# https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1

y_entrenamiento = to_categorical(entrenamiento[:, [24]])

early_stopping_monitor = EarlyStopping(patience=3)

model = Sequential()

model.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=24))
model.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(2, activation='softmax', kernel_initializer='random_normal'))

# model.add(Dense(25, activation='relu', input_shape=(1, )))
# model.add(Dense(25, activation='relu'))
# model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_entrenamiento, y_entrenamiento, epochs=5, callbacks = [early_stopping_monitor])

#en comentarios, los valores REALES
# 1
x_predict = x_entrenamiento[23]
some_test = [x_predict]
x_predict = np.array(some_test)
predictions0 = model.predict(x_predict)
# 1
x_predict = x_entrenamiento[24]
some_test = [x_predict]
x_predict = np.array(some_test)
predictions1 = model.predict(x_predict)
# 0 
x_predict = x_entrenamiento[25]
some_test = [x_predict]
x_predict = np.array(some_test)
predictions2 = model.predict(x_predict)
# 0
x_predict = x_entrenamiento[26]
some_test = [x_predict]
x_predict = np.array(some_test)
predictions3 = model.predict(x_predict)
print('breakpoint')