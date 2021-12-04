#import packages
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1.keras import metrics  
#tf.get_logger().setLevel('ERROR')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler # standardize features by removing the mean and scaling to unit variance.
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

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


print(train_X.shape)
print(test_X.shape)
print(train_label.shape)
print(test_label.shape)

batch_size = 128
epochs = 10
np.random.seed(1234)

# NN network

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


# binary classification where the target values are in the set {0, 1}
fashion_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])

print("-------model summary------------")
fashion_model.summary()
print("-------starting to train the model ------------")

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
plt.show()
plt.savefig('Training and validation accuracy.png')


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss', color='r')
plt.title('Training and validation loss')
plt.legend()
plt.figure()
plt.show()
plt.savefig('Training and validation loss.png')


print("----------Confusion Matrix and Classification Report-----------")
y_pred = fashion_model.predict(valid_X)
conf_matrix = confusion_matrix(y_true = valid_X, y_pred = y_pred)

# Print the confusion matrix using Matplotlib

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
plt.savefig('rf_confusion_matrix.png')


### RF Accuracy Score

print("----------Accuracy Score----------------")
print(accuracy_score(valid_X, y_pred))

print("------------Classification Report----------")
target_names = ['fake', 'real']
print(classification_report(valid_X, y_pred, target_names=target_names))


# References
#- https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#- https://www.machinecurve.com/index.php/2020/04/05/how-to-find-the-value-for-keras-input_shape-input_dim/
#- https://github.com/adriangb/scikeras

