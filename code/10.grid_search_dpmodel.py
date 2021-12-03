#import packages
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1.keras import metrics  

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# Load the data

data = np.load("X_data.npy", allow_pickle=True)
#data = data.astype(np.float32)

print(data.shape)

fake = np.zeros((9720, 1))
real = np.ones((9750, 1))
label = np.concatenate((fake, real), axis = 0)
#label = np.concatenate((np.zeros(shape = (1, 9720)), np.ones(shape = (1, 9750))), axis = -1)
print(label.shape)

train_X, test_X, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=42)

np.random.seed(1234)

# NN network

def deepLearning_model():
    # create model
    fashion_model = tf.keras.Sequential()
    fashion_model.add(tf.keras.layers.Dense(8192, input_dim = len(train_X[1]), activation = 'relu')) # input_dim = one-dimensional flattened arrays,
    fashion_model.add(tf.keras.layers.Dropout(0.2))

    fashion_model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
    fashion_model.add(tf.keras.layers.Dropout(0.2))

    fashion_model.add(tf.keras.layers.Dense(1024))

    fashion_model.add(tf.keras.layers.Dense(512))

    fashion_model.add(tf.keras.layers.Dense(256, activation='softmax'))

    fashion_model.add(tf.keras.layers.Dense(64, activation='sigmoid'))

    fashion_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    # binary classification where the target values are in the set {0, 1}
    fashion_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return fashion_model

np.random.seed(1234)
model = KerasClassifier(build_fn = deepLearning_model, verbose=1)

# define the grid search parameters
batch_size = [64, 128, 256]
epochs = [50, 100, 200, 400]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(train_X, train_label)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

print(grid_result.best_estimator_)

