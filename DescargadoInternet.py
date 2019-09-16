import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Deep Learning Libraries
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

# Loading Data
data = pd.read_csv('dataset.csv')

from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

predictors = data.drop(['ID','default.payment.next.month'], axis=1).as_matrix()
predictors = StandardScaler().fit_transform(predictors)

target = to_categorical(data['default.payment.next.month'])

non_default = len(data[data['default.payment.next.month']==0])
default = len(data[data['default.payment.next.month']==1])
ratio = float(default/(non_default+default))
print('Default Ratio :',ratio)

n_cols = predictors.shape[1]
early_stopping_monitor = EarlyStopping(patience=2)
class_weight = {0:ratio, 1:1-ratio}

model = Sequential()
model.add(Dense(25, activation='relu', input_shape = (n_cols,)))
model.add(Dense(25, activation='relu'))
#model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(predictors, target, epochs=20, validation_split=0.3, callbacks = [early_stopping_monitor],class_weight=class_weight)

