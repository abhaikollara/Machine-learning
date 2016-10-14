
# coding: utf-8

# # MNIST Classification using CNNs

# Import required libraries

# In[5]:

print 'Importing libraries...'
import numpy as np
import pandas as pd
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# Import the datasets

# In[31]:
print 'Importing datasets...'
test=pd.read_csv('../datasets/MNIST/mnist_test_kaggle.csv', header=None, skiprows=1)
train=pd.read_csv('../datasets/MNIST/mnist_train_kaggle.csv', header=None, skiprows=1)


# Seperate features and labels

# In[41]:
print 'Preprocessing...'
train_labels = train[0].values
test_labels = test[0].values
train_features = train.drop([0], axis=1).values.astype('float32')
test_features = test.drop([0], axis=1).values.astype('float32')

# Reshape the features

# In[45]:

train_features = train_features.reshape(train_features.shape[0],1,28,28)
test_features = test_features.reshape(test_features.shape[0],1,28,28)

# Normalise the variables

# In[46]:
print 'Normalising...'
train_features = train_features/float(255)
test_features = test_features/float(255)


# One hot encode the labels

# In[47]:
print 'One hot encoding...'
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)


# Define model

# In[52]:
print 'Defining model...'
def Conv_model():
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def larger_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# In[ ]:
print 'Builiding model...'
# build the model
model = larger_model()
# Fit the model
print 'Training model...'
startTime = time.time()
model.fit(train_features, train_labels, validation_data=(test_features, test_labels), nb_epoch=10, batch_size=200, verbose=2)
endTime = time.time()
# Final evaluation of the model
scores = model.evaluate(test_features, test_labels, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

print "Saving model..."
model.save("largeConvolutionalKaggle.h5")
totalSeconds = int(endTime - startTime)
print "Time taken : ",int(totalSeconds/60),'minutes',int(totalSeconds%60),'seconds'
