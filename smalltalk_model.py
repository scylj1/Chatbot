'''

Train a three layer sequential model for small talk

'''

import json
import keras
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sentiment_data import get_document, tfidf_weighting
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore warnings

classes = []
documents = []
vocabulary = []
labels = []

# load corpus
data_file = open('data/smalltalk/intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        document = get_document(pattern)
        documents.append(document)
        labels.append(intent['tag'])
        for item in document:
            if item not in vocabulary:
                vocabulary.append(item)

# get parameters
N = len(documents)
n = np.zeros(len(vocabulary))
for voc in vocabulary:
    index = vocabulary.index(voc)
    for doc in documents:
        if voc in doc:
            n[index] += 1

# Create bag-of-words input and output
bow = []
out_put = []
y0 = [0] * len(classes)

for document in documents:
    index = documents.index(document)
    y = list(y0)
    y[classes.index(labels[index])] = 1
    out_put.append(y)
    
    vector = np.zeros(len(vocabulary))
    for item in document:
        index = vocabulary.index(item)
        vector[index] += 1
    bow.append(tfidf_weighting(vector, n, N))

# save data
pickle.dump(vocabulary,open('data/smalltalk/words.pkl','wb'))
pickle.dump(classes,open('data/smalltalk/classes.pkl','wb'))
pickle.dump(n,open('data/smalltalk/n.pkl','wb'))
pickle.dump(N,open('data/smalltalk/documents.pkl','wb'))

# create train and test sets
train_x = list(bow)
train_y = list(out_put)

# create sequential model - 2 layers 
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#keras.utils.vis_utils.plot_model(model, "model/sequential.png", show_shapes=True, dpi=1440)
# compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fit model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)

# save model
model.save('model/smalltalk_model')

