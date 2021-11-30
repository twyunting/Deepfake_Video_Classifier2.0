#import packages
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1.keras import metrics

import matplotlib.pyplot as plt
import numpy as np

# Load the data

data = np.load("X_data.npy", allow_pickle=True)
data = data.astype(np.float32)

print(data.shape)

fake = np.zeros((9720, 1))
real = np.ones((9750, 1))
label = np.concatenate((fake, real), axis = 0)
#label = np.concatenate((np.zeros(shape = (1, 9720)), np.ones(shape = (1, 9750))), axis = -1)
print(label.shape)

train_X, test_X, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=42)


print(train_X.shape)
print(test_X.shape)
print(train_label.shape)
print(test_label.shape)

batch_size = 8
epochs = 10
np.random.seed(1234)

# NN network

fashion_model = tf.keras.Sequential()
fashion_model.add(tf.keras.layers.Dense(8192, input_dim = len(train_X[1]), activation = 'relu')) # input_dim = one-dimensional flattened arrays,
fashion_model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
fashion_model.add(tf.keras.layers.Dense(1024))
fashion_model.add(tf.keras.layers.Dense(512))
fashion_model.add(tf.keras.layers.Dense(256, activation='softmax'))
fashion_model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
fashion_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# binary classification where the target values are in the set {0, 1}
fashion_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])

print("-------model summary------------")
fashion_model.summary()
print("-------starting to train the model ------------")
#data = tf.convert_to_tensor(data, dtype=tf.float32)
fashion_train = fashion_model.fit(train_X, train_label,
                                  batch_size=batch_size,epochs=epochs,
                                  verbose=1,validation_data=(test_X, test_label))

print("------------------Training Outcomes----------------------")
# evaluate the keras model
train_eval = fashion_model.evaluate(train_X, train_label)
print('Training loss:', train_eval[0])
print('Traning accuracy:', train_eval[1])

print("------------------Testing Outcomes----------------------")
# evaluate the keras model
test_eval = fashion_model.evaluate(test_X, test_label)
print('Testing loss:', test_eval[0])
print('Testing accuracy:', test_eval[1])

print("------------------Graphs----------------------")
accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy', color='r')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.savefig('Training and validation accuracy.png')


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss', color='r')
plt.title('Training and validation loss')
plt.legend()
plt.figure()
#plt.show()
plt.savefig('Training and validation loss.png')

# References
#- https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#- https://www.machinecurve.com/index.php/2020/04/05/how-to-find-the-value-for-keras-input_shape-input_dim/
#- https://github.com/adriangb/scikeras

