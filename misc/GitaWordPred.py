
# coding: utf-8

# In[12]:

# from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.preprocessing.text import *
from keras.models import load_model
import numpy as np
import random
import sys


# In[31]:

text = open('input.txt').read().lower()
words = text_to_word_sequence(text)

vocab = sorted(list(set(words)))
print('total words:', len(vocab))
word_indices = dict((w, i) for i, w in enumerate(vocab))
indices_word = dict((i, w) for i, w in enumerate(vocab))


# In[28]:

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 5
step = 3
sentences = []
next_word = []
for i in range(0, len(words) - maxlen, step):
    sentences.append(words[i: i + maxlen])
    next_word.append(words[i + maxlen])
print('nb sequences:', len(sentences))


# In[36]:

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(vocab)), dtype=np.bool)
y = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_word[i]]] = 1
print (y.shape)
print (X.shape)


# In[ ]:

# build the model: a single LSTM

print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(vocab))))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 120):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    # model=model = load_model('gita.h5')
    model.fit(X, y, batch_size=128, nb_epoch=1)
    # model.save('gita.h5')
    start_index = random.randint(0, len(words) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('-' * 50)
        print('Diversity:', diversity)

        generated = []
        sentence = words[start_index: start_index + maxlen]
        generated.append(sentence)
        print(' Generating with seed:')
        print (sentence)
        print('-' * 50)
        print()
        # sys.stdout.write(generated)

        for i in range(100):
            x = np.zeros((1, maxlen, len(vocab)))
            for t, word in enumerate(sentence):
                x[0, t, word_indices[word]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            generated.append(next_word)
            sentence = sentence[1:]
            sentence.append(next_word)
            
            sys.stdout.write(' ')
            sys.stdout.write(next_word)
            sys.stdout.flush()
        print()

