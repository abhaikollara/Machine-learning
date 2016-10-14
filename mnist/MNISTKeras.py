
# coding: utf-8

# # MNIST Classification using Keras
# Import required libraries

# In[1]:

print "Importing libraries..."
import matplotlib.pyplot as plt
import pandas as pd
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


# Import datasets

# In[2]:

print 'Reading test data...'
test = pd.read_csv('../datasets/MNIST/mnist_test_small.csv', header=None)
print 'Reading training data...'
train = pd.read_csv('../datasets/MNIST/mnist_train_small.csv', header=None)


# Seperate labels and features and convert to numpy arrays

# In[19]:
print 'Creating and seperating features from labels...'
train_labels = train[0].values
test_labels = test[0].values
train_features = train.drop([0], axis=1).values.astype('float32')
test_features = test.drop([0], axis=1).values.astype('float32')


# Normalise the pixel values

# In[11]:
print 'Normalizing values...'
train_features = train_features/float(255)
test_features = test_features/float(255)


# One hot encode the labels

# In[20]:
print 'One hot encoding...'
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)


# Define baseline model

# In[26]:
def baseline_model():
    model = Sequential()
    model.add(Dense(784, input_dim=784, init='normal', activation='relu'))
    model.add(Dense(10, init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[28]:

print 'Building the model...'
# build the model
model = baseline_model()
# Fit the model
print 'Training model....'
startTime = time.time()
# model.fit(train_features, train_labels, validation_data=(test_features, test_labels), nb_epoch=10, batch_size=200, verbose=2)
model.fit(train_features, train_labels, validation_data=(test_features, test_labels), nb_epoch=10, batch_size=200, verbose=2)
endTime = time.time()
# Final evaluation of the model
scores = model.evaluate(test_features, test_labels, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

print "Training time : ",round((endTime-startTime),2)

