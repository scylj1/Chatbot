'''

Pre-process review data for sentiment model

'''

import numpy as np
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import pickle
from math import log10
from nltk.stem import SnowballStemmer
import json
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore warnings

'''
Load sentiment data

'''
def get_sentiment_data():
    # load corpus
    positive = []
    negative = []
    data_file = open('data/review/healths.json')
    for  i in range (0, 700):
        line = data_file.readline()
        intents = json.loads(line)
        score = intents['overall']
        if float(score) == 5.0:
            positive.append([intents['reviewText'], 1])
    print(len(positive))
    
    for  i in range (0, 10000):
        line = data_file.readline()
        intents = json.loads(line)
        score = intents['overall']
        if float(score) == 1.0:
            negative.append([intents['reviewText'], 0])
    print(len(negative))
    
    data_file.close()
    return negative+positive
            
                
'''
Tokenize, remove stopwords and get stemming

    Argument: string: input sentence
                stop: decide whether remove stopwords
            stemming: decide whether do stemming
     
    Returns: processed vector
'''
def get_document(string, stop = True, stemming=True):
 
    # Tokenise and remove punctuation
    tok_document = word_tokenize(string)
        
    if stop is True:
    # Remove stop words and normalise casing
        english_stopwords = stopwords.words('english')
        document = [word.lower() for word in tok_document if word.lower() not in english_stopwords]
    else:
        document = [word.lower() for word in tok_document]
                            
    # print(documents)
    if stemming is True:
    # Stemming
        sb_stemmer = SnowballStemmer('english')
        stemmed_document = [sb_stemmer.stem(word) for word in document]
        document = stemmed_document
        
    return document

'''
Tf-idf weighting method

    Argument: vector: original vector
                   n: number of documents contain the word
                   N: number of total documents
     
    Returns: processed vector
'''
def tfidf_weighting(vector, n, N):
    tfidf_vector = []
    i = 0
    for frequency in vector:
        n = np.array(n)  
        if n[i] != 0:
            tfidf_vector.append(log10(1+frequency)*log10(N/n[i]))   
        else:
            tfidf_vector.append(0)
        i+=1
    return np.array(tfidf_vector)

'''
Load data

    Argument: raw review data in a list
     
    Returns: processed data for words, classes, punctuation
'''
def load_data(text):
    
    # load data
    # get words and punctuations      
    x_texts = text            
    x_words = []
    x_punc = []
    
    for i in range(len(x_texts)):
        sentence = x_texts[i]
        cleaned_tokens = get_document(sentence)
        
        words = []
        punc = []
        for j in range(len(cleaned_tokens)):
            punc.append(cleaned_tokens[j])
            if cleaned_tokens[j] not in string.punctuation:
                words.append(cleaned_tokens[j])
        
        x_words.append(words)
        x_punc.append(punc)
     
    # load data for getting class
    f=open('data/review/positive_review.txt')
    pos_comment = []
    for line in f:
        pos_comment.append(line.strip())
    pos_comment.pop(0)
    pos_comment.pop(0)
        
    f=open('data/review/positive_emotion.txt')
    pos_emotion = []
    for line in f:
        pos_emotion.append(line.strip())
    pos_emotion.pop(0)
    pos_emotion.pop(0)
       
    f=open('data/review/negative_review.txt')
    neg_comment = []
    for line in f:
        neg_comment.append(line.strip())
    neg_comment.pop(0)
    neg_comment.pop(0)
        
    f=open('data/review/negative_emotion.txt')
    neg_emotion = []
    for line in f:
        neg_emotion.append(line.strip())
    neg_emotion.pop(0)
    neg_emotion.pop(0)
        
    f=open('data/review/opinion.txt')
    txt = []
    for line in f:
        txt.append(line.strip())
        
    perception = txt[3:24]
    regard = txt[26:]
        
    f=open('data/review/degree.txt')
    txt = []
    for line in f:
        txt.append(line.strip())        
    most = txt[3:67]
    very = txt[69:94]
    more = txt[96:118]
    slightly = txt[120:135]
    insufficiently = txt[137:148]
    over = txt[150:183]
    
    # set class
    word_class = [pos_comment, pos_emotion, neg_comment, neg_emotion, perception, regard, 
                 most, very, more, slightly, insufficiently, over]
    word_class_name = ['positive', 'positive', 'negative', 'negative', 'perception', 'regard', 
                 'most', 'very', 'more', 'slightly', 'insufficiently', 'over']
    
    # get class
    x_class = []   
    for a in range(len(x_words)):
        classlist = []
        for b in range(len(x_words[a])):
            flag = False
            for c in range(len(word_class)):
                for d in range(len(word_class[c])):
                    if x_words[a][b] == word_class[c][d]:
                        classlist.append(word_class_name[c])
                        flag = True
                        break
                if flag == True:
                    break
            if flag == False:
                classlist.append(',')
        x_class.append(classlist)   

    return x_class, x_words, x_punc

'''
Build up vocabulary

    Argument: classes, words and punctuation
     
    Returns: vocabulary, number of documents contain the word, number of total documents
'''
def build_vocabulary(x_class, x_words, x_punc):
    # build vocabulary
    vocabulary = []
    for document in x_class:
        for item in document:
            if item not in vocabulary:
                vocabulary.append(item)
    for document in x_words:
        for item in document:
            if item not in vocabulary:
                vocabulary.append(item)
    for document in x_punc:
        for item in document:
            if item not in vocabulary:
                vocabulary.append(item)
    
    N = len(x_texts)
    n = np.zeros(len(vocabulary))
    for voc in vocabulary:
        index = vocabulary.index(voc)
        for doc in x_texts:
            if voc in doc:
                n[index] += 1
    
    pickle.dump(vocabulary,open('data/review/vocabulary.pkl','wb'))
    pickle.dump(n,open('data/review/n.pkl','wb'))
    pickle.dump(N,open('data/review/documents.pkl','wb'))
    
    return vocabulary, n, N

'''
Process data for training

    Argument: classes, words and punctuation, vocabulary, 
              number of documents contain the word, number of total documents
     
    Returns: processed data for training
''' 
def get_training(x_class, x_words, x_punc, vocabulary, n, N):
    # get bag-of-words
    x_tokens_class = []
    for document in x_class:
        vector = np.zeros(len(vocabulary))
        for item in document:
            try:
                index = vocabulary.index(item)
                vector[index] += 1
            except ValueError:
                continue
        x_tokens_class.append(tfidf_weighting(vector, np.array(n), N))
    
    x_tokens_words = []
    for document in x_words:
        vector = np.zeros(len(vocabulary))
        for item in document:
            try:
                index = vocabulary.index(item)
                vector[index] += 1
            except ValueError:
                continue
        x_tokens_words.append(tfidf_weighting(vector, np.array(n), N))
        
    x_tokens_punc = []
    for document in x_punc:
        vector = np.zeros(len(vocabulary))
        for item in document:
            try:
                index = vocabulary.index(item)
                vector[index] += 1
            except ValueError:
                continue
        x_tokens_punc.append(tfidf_weighting(vector, np.array(n), N))
       
    # Define longest length   
    max_words_len = 250        
    max_punc_len = 300
    
    # formated data to training type    
    x_tokens_class = pad_sequences(x_tokens_class, maxlen=max_words_len,
                                   padding='pre', truncating='pre')
    x_tokens_words = pad_sequences(x_tokens_words, maxlen=max_words_len,
                                   padding='pre', truncating='pre')
    
    
    x_tokens_punc = pad_sequences(x_tokens_punc, maxlen=max_punc_len,
                                  padding='pre', truncating='pre')
      
    x_tokens_class.shape
    x_tokens_words.shape
    x_tokens_punc.shape
      
    data_dim = 300
    x_class_input = np.zeros([x_tokens_class.shape[0], x_tokens_class.shape[1], data_dim])
    for i in range(x_tokens_class.shape[0]):
        for j in range(x_tokens_class.shape[1]):
            x_class_input[i][j] = x_tokens_class[i][j]    
    
    x_words_input = np.zeros([x_tokens_words.shape[0], x_tokens_words.shape[1], data_dim])
    for i in range(x_tokens_words.shape[0]):
        for j in range(x_tokens_words.shape[1]):
            x_words_input[i][j] = x_tokens_words[i][j]    
    
    x_punc_input = np.zeros([x_tokens_punc.shape[0], x_tokens_punc.shape[1], data_dim])
    for i in range(x_tokens_punc.shape[0]):
        for j in range(x_tokens_punc.shape[1]):
            x_punc_input[i][j] = x_tokens_punc[i][j]
       
    x_class_input.shape
    x_words_input.shape
    x_punc_input.shape
    x_wordsclass_input = np.concatenate([x_words_input, x_class_input], axis=2)
    x_wordsclass_input.shape

    return x_wordsclass_input, x_punc_input

if __name__ == '__main__':
 
    # load data
    data = get_sentiment_data()
    random.shuffle(data)
    x_texts = []
    labels = []
    for d in data:
        x_texts.append(d[0])
        labels.append(d[1])
    # processed data
    x_class, x_words, x_punc = load_data(x_texts)
    vocabulary, n, N = build_vocabulary(x_class, x_words, x_punc)
    x_wordsclass_input, x_punc_input = get_training(x_class, x_words, x_punc, vocabulary, n, N)
    
    # save
    np.save('data/review/x_wordsclass_input.npy', x_wordsclass_input)
    np.save('data/review/x_punc_input.npy', x_punc_input)
    np.save('data/review/labels.npy', labels)