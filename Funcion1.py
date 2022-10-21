#------------------------------------------------------------------------------
#----------------------- Homework 3.1.a) --------------------------------------
#RNA that produces the function "3Sin(pi*x)" in the interval of [-1,1]. 
#Graph the solution of the network together with the graph of the function.
#------------------------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from numpy import asarray
from matplotlib import pyplot
import numpy as np
import re
np.set_printoptions(suppress=True)

x = asarray([i/500 for i in range(-500,500)])
y = asarray([3*np.sin(np.pi*i)  for i in x]) #Defined function

x = x.reshape((len(x), 1))
y = y.reshape((len(y), 1))

scale_x = MinMaxScaler()
x = scale_x.fit_transform(x)
scale_y = MinMaxScaler()
y = scale_y.fit_transform(y)

model = Sequential()
model.add(Dense(5, input_dim=1, activation='tanh', kernel_initializer='he_uniform'))
model.add(Dense(10, input_dim=1, activation='tanh', kernel_initializer='he_uniform'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam') #Optimizizer Adam
model.fit(x, y, epochs=100, batch_size=10, verbose=1)
yhat = model.predict(x)

# inverse transforms
x_plot = scale_x.inverse_transform(x)
y_plot = scale_y.inverse_transform(y)
yhat_plot = scale_y.inverse_transform(yhat)

# plot x vs yhat
pyplot.scatter(x_plot,y_plot, color='blue', label='Solución Actual')
pyplot.scatter(x_plot,yhat_plot,color='yellow', label='Solución predecida')
pyplot.title(r'$y = 3sin(\pi x)$')
pyplot.xlabel('Entrada x')
pyplot.ylabel('Salida y')
pyplot.legend()
pyplot.show()
