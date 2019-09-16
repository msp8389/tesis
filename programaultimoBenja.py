# importamos las librerias generales
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
# librerias para deep learning
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from keras.layers import Dropout



# carga de datos
datos = pd.read_csv('dataset.csv')
predict = pd.read_csv('predict2.csv')

entrenamiento = datos.values
# Elimina todos las columnas en las posiciones 0 y 24
sc = StandardScaler()
x_entrenamiento = entrenamiento[:, 1:24]
x_entrenamiento = sc.fit_transform(x_entrenamiento)

# https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1
# [0,1] = 1(paga)  y [1, 0] = 0(no paga)
y_entrenamiento = to_categorical(entrenamiento[:, 24])
early_stopping_monitor = EarlyStopping(patience=3, monitor='acc', mode='max')

model = Sequential()

model.add(Dense(32, kernel_initializer='random_uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, kernel_initializer='random_uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, kernel_initializer='random_uniform', activation='softmax'))


  
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.02), metrics=['accuracy'])

model.fit(x_entrenamiento, y_entrenamiento, epochs=100,
          callbacks=[early_stopping_monitor])

print("Evaluacion del modelo")
loss, acc = model.evaluate(x_entrenamiento, y_entrenamiento)
print ("Perdida: ",loss,"Accuracy: " ,acc)

# en comentarios, los valores REALES
# [0,1] = 1(paga)  y [1, 0] = 0(no paga)
print("[0,1] = 1(paga)","[1, 0] = 0(no paga)")
index_y = 0
cantidad_a_testear = 1
for predict in x_entrenamiento[:cantidad_a_testear]:
    some_test = [predict]
    x_predict = np.array(some_test)
    predictions = model.predict(x_predict)
    if (predictions[0][0] > predictions[0][1]):
        prediccion = "[1. 0.]"
    else:
        prediccion = "[0. 1.]"
    if (str(y_entrenamiento[index_y]) == prediccion):
        mas_info = "PRED CORRECTA"
    else:
        mas_info = "PRED INCORRECTA"
    print(predictions[0], 'PRED:', prediccion, 'VALOR:', y_entrenamiento[index_y], mas_info)
    index_y = index_y + 1
