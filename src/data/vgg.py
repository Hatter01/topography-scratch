import keras
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from data_loader import load_dataset
import numpy as np

#DATASET
X, y =  load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

#NETWORK
model = Sequential()
model.add(Conv2D(input_shape=(480,640,3), filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))


model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

import tensorflow as tf

def loss_funtion(real_eps, eps_new):
    x = tf.abs(real_eps - eps_new)
    return tf.math.minimum(x, 1-x)


def my_metric_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

compare=loss_funtion(np.array([0,1,0.5, 0.25]), np.array([1,0.75, 1, 0.5]))
print(compare.numpy(), "should be: [0.   0.25 0.5  0.25]")

#COMPILATION
opt = Adam(learning_rate=0.01)# optimizer
model.compile(optimizer=opt, loss=loss_funtion, metrics=[keras.metrics.MeanSquaredError()])
model.summary()

#TRAIN THE MODEL
hist = model.fit(X_train, y_train, epochs=5, 
                    validation_data=(X_test, y_test))