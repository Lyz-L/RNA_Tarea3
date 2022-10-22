#------------------------------------------------------------------------------
#----------------------- Homework 3.1.b) --------------------------------------
#RNA that produces the function "xy' + y = x^2 * cosx" with y(0)=0 in 
#the interval of [-5,5].
#Graph the solution of the network together with the graph of the function.---
#------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np

class ODEsolver(Sequential):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.loss_tracker = keras.metrics.Mean(name = 'my_loss')

  @property
  def metrics(self):
    return [self.loss_tracker]

  def train_step(self, data):
    batch_size = tf.shape(data)[0]
    x = tf.random.uniform((batch_size, 1), minval = -5, maxval = 5)

    with tf.GradientTape() as tape:
      with tf.GradientTape() as tape2:
        tape2.watch(x)
        y_pred = self(x, training = True)
      dy = tape2.gradient(y_pred, x)
      x_o = tf.zeros((batch_size, 1))
      y_o = self(x_o, training = True)
      eq = x*dy + y_pred - x**2 * tf.math.cos(x)
      ic = y_o 
      loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic)

    grads = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    self.loss_tracker.update_state(loss)

    return {'my_loss': self.loss_tracker.result()}

model = ODEsolver()

model.add(Dense(15, activation='tanh', input_shape=(1,)))
model.add(Dense(12, activation='relu')) #tanh
model.add(Dense(1, activation='tanh'))
#model.add(Dense(10, activation='tanh'))
#model.add(Dense(15, activation='tanh'))

model.summary()

model.compile(optimizer='adam',metrics=['loss']) #RMSprop
x=tf.linspace(-5,5,100)
history = model.fit(x,epochs=100, verbose=1) #100

x_testv = tf.linspace(-5,5,100)
a=model.predict(x_testv)
plt.plot(x_testv,a, color='g', label='Solución predecida')
plt.plot(x_testv,(((x*x -2)*np.sin(x))/x) + 2*np.cos(x), color='r', label='Solución real')
plt.title(r'Ecuación Diferencial')
plt.xlabel('Entrada x')
plt.ylabel('Salida y')
plt.legend()
plt.show()
