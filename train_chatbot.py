import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.leads(data_file)

for intent in intents['intents']:
	for pattern in intents['patterns']:

		# tokenize each word
		w = nltk.word_tokenize(pattern)
		words.extend(w)
		# add documents in the corpus
		documents.append((w, intent['tag']))

		# add to our class list
		if intent['tag'] not in classes:
			classes.append(intent['tag'])

# lemmatize, lower each word and remove duplicates

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

