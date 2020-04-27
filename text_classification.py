from _ast import keyword

import tensorflow as tf
from dateutil.parser import _resultbase
from scipy.special.cython_special import ker
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

## TODO - revert numpy to 1.16.1 duo to faluare in indexing, known issue for numpy.
## TODO - test with latest numpy version  (1.18.3) - later descover that on numpy open issues table.
from tensorflow_core.python import variable_scope

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000) #leave out erelevet words.

print(train_data[0])

#word_index = imdb.get_word_index()
word_index = data.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0;
word_index["<START>"] = 1;
word_index["<UNK>"] = 2;
word_index["<UNUSED>"] = 3;

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#need to determent a definite legnth for all of our data (using some arbitray limit)
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


# decode to human view

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# model is here!, our purpase of this model is to get a GOOD or BAD review in the end of it.
# in order to do so we only one output neuron which represent 0 OR 1

model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16)) # creating a vector grid - for every word open a 16 neurons layers
model.add(keras.layers.GlobalAveragePooling1D()) # normalizing the vector grid (embedding layer), by avg..
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid")) # a function that returns a ALMOST logical output based on his value like (0.3).

model.summary()

#tain and use the model!

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) # loss will trim the model output to a valid binary result.
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val =train_labels[:10000]
y_train = train_labels[10000:]

# fit the model:

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

result = model.evaluate(test_data, test_labels)


#some utils functions
def pretty():
    test_review = test_data[0]
    predict = model.predict([test_review])
    print(f"Review: {decode_review(test_review)} prediction: {str(predict[0])} Actual! {str(test_labels[0])}")
    print(result)

pretty()




