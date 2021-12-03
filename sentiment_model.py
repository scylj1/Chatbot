'''

Train a double layer bidirectional GRU model for sentiment classification

'''

import numpy as np
import keras
from keras.layers import GRU
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore warnings

# load data
x_wordsclass_input = np.load('data/review/x_wordsclass_input.npy')
x_punc_input = np.load('data/review/x_punc_input.npy')
labels = np.load('data/review/labels.npy')
max_words_len = 250
max_punc_len = 300
data_dim = 300

# get labels
y = np.zeros([993, 1])
y.shape
for  i in range (0, 993):
    y[i][0] = labels[i]
  
# model settings
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-4, patience=0, verbose=1)
callbacks = [earlystopping, lr_reduction]

# Double layer bidirectional GRU model
input1 = keras.layers.Input(shape=(max_words_len, 2*data_dim), name='input_wordsclass')
x11 = keras.layers.Bidirectional(GRU(32, return_sequences=True))(input1)
x12 = keras.layers.Bidirectional(GRU(32))(x11)
input2 = keras.layers.Input(shape=(max_punc_len, data_dim), name='input_wordspunc')
x21 = keras.layers.Bidirectional(GRU(32, return_sequences=True))(input2)
x22 = keras.layers.Bidirectional(GRU(32))(x21)
con = keras.layers.concatenate([x12, x22])
y1 = keras.layers.Dense(32)(con)
y2 = keras.layers.Dropout(0.5)(y1)
output = keras.layers.Dense(1, name='output', activation='sigmoid')(y2)
model = keras.models.Model(inputs=[input1, input2], outputs=output)
model.summary()

# keras.utils.vis_utils.plot_model(model, "model/gru.png", show_shapes=True, dpi=1440)
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['binary_accuracy'])

# fit model
history = model.fit({'input_wordsclass':x_wordsclass_input, 'input_wordspunc':x_punc_input},
                    {'output':y},
                    batch_size=16, epochs=20,
                    validation_split=0.2,
                    callbacks=callbacks)

# save model
model.save('model/sentiment_model')
