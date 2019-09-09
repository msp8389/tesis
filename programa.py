# importamos las librerias generales 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# carga de datos
datos = pd.read_csv('dataset.csv')
predict = pd.read_csv('predict2.csv')
#mostramos los metadatos de la informacion para verificar que se hayan cargado correctamente
datos.info()
print(datos.describe())


#dividimos los datos en entrenamiento (%83.44) y testeo (%16.66) 
#utilizamos rand pero hay otros tipos de random como randn y randint
msk = np.random.rand(len(datos)) < 0.9
entrenamiento = datos[msk]
testeo = datos[~msk]



from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

#eliminamos la columna id y la etiqueta. Axis = 1 significa que se recorre por columna, en vez de por fila (Axis = 0)
x_entrenamiento = entrenamiento.drop(['ID','default.payment.next.month'], axis=1).as_matrix()
#Transformamos los datos con la funcion de sklearn 
x_entrenamiento = StandardScaler().fit_transform(x_entrenamiento)
#convierte un vector de clase (enteros) en una matriz de clase binaria
y_entrenamiento = to_categorical(entrenamiento['default.payment.next.month'])

x_testeo = testeo.drop(['ID','default.payment.next.month'], axis=1).as_matrix()
x_testeo = StandardScaler().fit_transform(x_testeo)
y_testeo = to_categorical(testeo['default.payment.next.month'])



print(x_testeo)

# librerias para deep learning
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight



# calculo 
non_default = len(entrenamiento[entrenamiento['default.payment.next.month']==0])
default = len(entrenamiento[entrenamiento['default.payment.next.month']==1])
ratio = float(default/(non_default+default))
print('Default Ratio :',ratio)




n_cols = x_entrenamiento.shape[1]
early_stopping_monitor = EarlyStopping(patience=2)
class_weight = {0:ratio, 1:1-ratio}

model = Sequential()
model.add(Dense(25, activation='relu', input_shape = (n_cols,)))
model.add(Dense(25, activation='relu'))
#model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_entrenamiento, y_entrenamiento, epochs=20, callbacks = [early_stopping_monitor],class_weight=class_weight)

model.evaluate(x_testeo, y_testeo)

print(predict)

model.predict(predict)
#print(predictors)
