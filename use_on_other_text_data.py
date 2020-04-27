from _ast import keyword

import tensorflow as tf
from dateutil.parser import _resultbase
from scipy.special.cython_special import ker
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_core.python import variable_scope

#TODO - trim un wanted new lines in the imported text file.
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# print(train_data[0])
word_index = data.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0;
word_index["<START>"] = 1;
word_index["<UNK>"] = 2;
word_index["<UNUSED>"] = 3;

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


# decode to human view

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


model = keras.models.load_model("my_model.h5")


# result = model.evaluate(test_data, test_labels)


def reivew_encode(s):
    encoded = [1];

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


with open("review_3_5_2", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(".", "").replace(",", "").replace("(", "").replace(")", "").replace(":", "").replace("\"",
                                                                                                                  "").strip().split(
            " ")
        encode = reivew_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
                                                            maxlen=250)  # trim the data to our model border (250 words)
        predict = model.predict(encode)
        print(line)
        print(encode)  # just print my encoded text
        print(predict[0])  # output of model


# some utils functions
def pretty():
    test_review = test_data[0]
    predict = model.predict([test_review])
    print(f"Review: {decode_review(test_review)} prediction: {str(predict[0])} Actual! {str(test_labels[0])}")
    print(result)
