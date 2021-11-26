

# implement pip as a subprocess:
#import sys
#import subprocess

#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])

# Packages
#import numpy as np
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.layers import Sequential, Input, Model
#from tensorflow.keras.optimizers import Adam

#import tensorflow
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1 import layers

"""
# Load the data

data = np.load("X_data.npy", allow_pickle=True)
data = data.astype(np.float32)
#data = tf.convert_to_tensor(data, dtype=tf.float32)

print(data.shape)

fake = np.zeros((9720, 1))
real = np.ones((9750, 1))
label = np.concatenate((fake, real), axis = 0)
print(label.shape)

train_X, test_X, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=13)

print(train_X.shape)
print(test_X.shape)
print(train_label.shape)
print(test_label.shape)

fashion_model = Sequential()
fashion_model.add(Dense(64, input_dim = len(train_X[1]), activation='relu')) # input_dim = one-dimensional flattened arrays,
fashion_model.add(Dense(128, activation='sigmoid'))
fashion_model.add(Dense(1, activation='softmax'))
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])
fashion_model.summary()
#data = tf.convert_to_tensor(data, dtype=tf.float32)
fashion_train = fashion_model.fit(train_X, train_label,
                                  batch_size=batch_size,epochs=epochs,
                                  verbose=1,validation_data=(test_X, test_label))

print("------------------best estimator----------------------")
# evaluate the keras model
_, accuracy = fashion_model.evaluate(train_X, train_label)
print('Accuracy: %.2f' % (accuracy*100))

print("------------------best paprmeters----------------------")
# evaluate the keras model
test_eval = fashion_model.evaluate(test_X, test_label, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = fashion_train.history['accuracy']
val_accuracy = fashion_train.history['val_accuracy']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy', color='r')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss', color='r')
plt.title('Training and validation loss')
plt.legend()
plt.show()
plt.savefig('10.confusion_matrix.png')
"""
# References
#- https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#- https://www.machinecurve.com/index.php/2020/04/05/how-to-find-the-value-for-keras-input_shape-input_dim/
#- https://github.com/adriangb/scikeras

