"""

Main function for Chatbot
Functions for Identification, Small talk, Question answering, Transaction, Review

"""
import nltk
import re
import numpy as np
import random
import csv
import json
import pickle
from nltk.corpus import stopwords
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore warnings
from scipy import spatial
from keras.models import load_model
from sentiment_data import get_document, tfidf_weighting, load_data, get_training


class Identification:
       
    '''
    Check and Save user's name
    '''    
    def identity(self):
        f = open('data/customer/common.txt','r',errors = 'ignore', encoding = 'utf-8') 
        words = []
        line = f.readline()
        while line: 
            words.append(line.replace("\n", ""))
            line = f.readline()
        f.close()
    
        f = open('data/customer/name.txt','r',errors = 'ignore', encoding = 'utf-8') 
        names = []
        line = f.readline()
        while line: 
            names.append(line.replace("\n", "").lower())
            line = f.readline()
        f.close()
    
        user_response = input("YOU: ")
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        tok_document = tokenizer.tokenize(user_response)

        name = []
        english_stopwords = stopwords.words('english')   
        name.append([word for word in tok_document if ((word.lower() in names) 
                                                          or (word.lower() not in english_stopwords+words))])    
        username = ""
        if name[0] == []:
            print("ROBO: Please type your name again to confirm. ")
            username = input("YOU: ")
        else:
            for word in name[0]:
                username += str(word) + " "
        print("ROBO: Hi, " + username + "!")
        
        f = open('data/customer/users.txt','a',errors = 'ignore', encoding = 'utf-8')
        f.writelines(username+'\n')
        f.close()
     
    '''
    Check if user enter something related to the name
    '''
    def is_name(self, input):
        document = get_document(input, stop = False, stemming = False)
        if 'name' in document or 'call' in document:
            if 'your' in document or 'you' in document:
                print("ROBO: Hi, my name is RoBo!")
                return True
            else:
                f = open('data/customer/users.txt','r',errors = 'ignore', encoding = 'utf-8')
                lines = f.readlines()  
                last_line = lines[-1]
                print("ROBO: Hi, " + last_line.strip())
                return True
        else:
            return False


class Smalltalk:
    
    model = []
    intents = []
    words = []
    classes = []
    n = []
    N = []
    predict = []
    
    def __init__(self):  
        self.model = load_model('model/smalltalk_model')  
        self.intents = json.loads(open('data/smalltalk/intents.json').read())
        self.words = pickle.load(open('data/smalltalk/words.pkl','rb'))
        self.classes = pickle.load(open('data/smalltalk/classes.pkl','rb'))
        self.n = pickle.load(open('data/smalltalk/n.pkl','rb'))
        self.N = pickle.load(open('data/smalltalk/documents.pkl','rb'))
    
    '''
    Get bag of words
    '''
    def get_bow(self, sentence, words):
        # tokenize
        sentence_words = get_document(sentence)
        # bag of words
        vector = np.zeros(len(words))
        for item in sentence_words:
            try:
                index = words.index(item)
                vector[index] += 1
            except ValueError:
                continue
        bag = tfidf_weighting(vector, self.n, self.N)
    
        return(np.array(bag))
    
    '''
    Predict the intent of user input
    '''
    def predict_class(self, sentence, model):
        bow = self.get_bow(sentence, self.words)
        if sum(bow) == 0:
            return []
        res = model(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.8
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list
    
    '''
    Get a random response from that intent
    '''
    def get_response(self, predict, intents_json):
        tag = predict[0]['intent']
        intents = intents_json['intents']
        for intent in intents:
            if(intent['tag'] == tag):
                result = random.choice(intent['responses'])
                break
        return result
    
    '''
    Get response for uer input
    '''
    def talk_response(self, userinput):
        self.predict = self.predict_class(userinput, self.model)
        if self.predict != []:
            response = self.get_response(self.predict, self.intents)
            return response
        else:
            return "Sorry, I do not understand"
        
             
class Information_retrieval:
    
    N = 0 # number of documents
    n = []
    pairs = [] # Q & A pairs
    vocabulary = []
    bow = []
    
    def __init__(self):     
        self.pairs = self.load_corpus()
        self.bow, self.vocabulary = self.get_bow(self.pairs)
    
    '''
    Load question answer corpus
    '''
    def load_corpus(self):
        pairs = []
        QAs = csv.reader(open('data/QA/QA_dataset.csv','r', errors = 'ignore', encoding = 'utf-8'))
        for qa in QAs:
            pairs.append([qa[1], qa[2]])  
        pairs.pop(0)
        return pairs
     
    '''
    Get bag of words
    '''
    def get_bow(self, pairs):
        documents = []
        for pair in pairs:
            documents.append(get_document(pair[0], stop=False))
            # Create vocabulary
        vocabulary = []
        for document in documents:
            for item in document:
                if item not in vocabulary:
                    vocabulary.append(item)
        self.N = len(documents)
        self.n = np.zeros(len(vocabulary))
        for voc in vocabulary:
            index = vocabulary.index(voc)
            for doc in documents:
                if voc in doc:
                    self.n[index] += 1

        bow = []
        for document in documents:
            vector = np.zeros(len(vocabulary))
            for item in document:
                index = vocabulary.index(item)
                vector[index] += 1
            bow.append(tfidf_weighting(vector, np.array(self.n), self.N))  
        return bow, vocabulary
    
    '''
    Check similarity for userinput with database
    '''
    def get_similarity_cosine(self, query, stop = True, stemming=True):    
        stemmed_query = get_document(query, stop = False)
        vector_query = np.zeros(len(self.vocabulary))
        for stem in stemmed_query:
            try:
                index = self.vocabulary.index(stem)
                vector_query[index] += 1
            except ValueError:
                continue
        vector_query = tfidf_weighting(vector_query, np.array(self.n), self.N)
    
        similarities = []
        if sum(vector_query) != 0:
            for vector in self.bow:
                similarities.append(1 - spatial.distance.cosine(vector, vector_query))      
        else:
            similarities.append(0)
        return similarities


class Transaction:
    
    item = ""
    address = ""
    phone = ""
    
    '''
    Ask user for mask type input
    '''
    def get_response(self):
        print("ROBO: We are saling 3 types of masks, N95 mask, Medical surgical mask and Activated carbon mask. Which one or more do you prefer?")
        flag=True
        while(flag==True):
            user_response = input("YOU: ")
            document = get_document(user_response, stemming = False)
            if(('n95' not in document) and ('medical' not in document) and ('surgical' not in document) 
                                   and ('carbon' not in document) and ('activated' not in document)):
                print("ROBO: Sorry, we only sale N95 mask, Medical surgical mask and Activated carbon mask")
                continue
            if('n95' in document):
                self.get_quantity("N95 mask")
                
            if('medical' in document or 'surgical' in document):
                self.get_quantity("Medical surgical mask")
            
            if('carbon' in document or 'activated' in document):
                self.get_quantity("Activated carbon mask")
            flag=False
            self.get_address()
    
    '''
    Get quantities
    '''
    def get_quantity(self, mask):
        print("ROBO: How many", mask, "do you want to buy? (Enter a number)")
        flag=True
        while(flag==True):
            user_response = input("YOU: ")
            number = re.findall(r'\b\d+\b', user_response)
            if number != []:
                print("ROBO: You ordered %d" %int(number[0]))
                self.item += "Type: " + mask + ", Quantity: " + number[0] + ", "
                flag = False
            else:
                print("ROBO: Invalid input, please enter a integer number")
    
    '''
    Get address
    '''
    def get_address(self):
        print("ROBO: Thanks! May I have your address please. ")
        user_response = input("YOU: ")
        self.address = user_response
        print("ROBO: Thanks. ")
        self.get_phone()
     
    '''
    Get phone numbers
    '''
    def get_phone(self):
        print("ROBO: May I have your phone number please. ")
        flag=True
        while(flag==True):
            user_response = input("YOU: ")
            phone = re.findall(r'\d+', user_response)
            if phone != []:
                print("ROBO: You phone number is %d" %int(phone[0]))
                self.phone = phone[0]
                self.save()
                flag = False               
            else:
                print("ROBO: Invalid input, please enter again")
    
    '''
    Save order information
    '''
    def save(self):
        f = open('data/customer/users.txt','r',errors = 'ignore', encoding = 'utf-8')
        lines = f.readlines()  
        last_line = lines[-1]
        name = last_line.strip()
        f.close()
        
        order = "Name: " + str(name) + ", " + self.item + "Phone: " + self.phone + ", Address: " + self.address
        print("ROBO: Thanks for ordering")
        print("Order information: ")        
        print(order)
        print("Your order is completed, we will deliver it to you as soon as possible.")
        print("ROBO: You can talk with me, ask me questions, make orders and reviews. If you want to exit, type Bye!")
        f = open('data/order/orders.txt','a',errors = 'ignore', encoding = 'utf-8')
        f.writelines(str(order) + "\n" )
        f.close()


class Review:
    model = []
    n = []
    N = []
    vocabulary = []
    
    def __init__(self):        
        self.vocabulary = pickle.load(open('data/review/vocabulary.pkl','rb'))
        self.n = pickle.load(open('data/review/n.pkl','rb'))
        self.N = pickle.load(open('data/review/documents.pkl','rb'))
        self.model = load_model('model/sentiment_model')
  
    '''
    Get the response for user review input
    '''
    def get_response(self):
        print("ROBO: Please give us your comment. ")      
        user_response = input("YOU: ")
        
        x_class, x_words, x_punc = load_data([user_response])        
        x_wordsclass_input, x_punc_input = get_training(x_class, x_words, x_punc, 
                                                        self.vocabulary, self.n, self.N)
        
        pre_result = self.model({'input_wordsclass':x_wordsclass_input, 'input_wordspunc':x_punc_input})
        result = float(pre_result[0][0])
        if (result > 0.8):
            print("ROBO: That's good! Thanks for your comment. ")
        else:
            print("ROBO: Thanks for your comment. We will get our best to improve it. ")
        print("ROBO: You can talk with me, ask me questions, make orders and reviews. If you want to exit, type Bye!")

if __name__ == '__main__':
    
    print("Wait for initializing") 
    nltk.download('stopwords')
    nltk.download('punkt')
    
    # initialize
    i = Identification()    
    ir = Information_retrieval()
    st = Smalltalk()
    t = Transaction()
    r = Review()
    
    print("ROBO: Hi, welcome to Mask Sale System! My name is Robo. May I have your name please?")    
    i.identity()
    
    print("ROBO: You can talk with me, ask me questions, make orders and reviews. If you want to exit, type Bye! ")
       
    flag=True
    while(flag==True):
        user_response = input("YOU: ")
        user_response = user_response.lower()
        if(user_response!='bye'):
            if(i.is_name(user_response)):
                continue;
            else:
                similarities = ir.get_similarity_cosine(user_response, stop = False, stemming=False)
                max_similar = max(similarities)
                if (max_similar > 0.8):
                    max_index = similarities.index(max_similar)
                    print("ROBO: " + ir.pairs[max_index][1])
                else:  
                    response = st.talk_response(user_response)                                                              
                    if response == "Navigating to transaction mode":
                        print("ROBO: " + response)
                        t.get_response()
                    elif  response == "Navigating to review mode":
                        print("ROBO: " + response)
                        r.get_response()
                    else:
                        print("ROBO: " + response)
                        
        else:
            flag=False
            print("ROBO: Thanks for your time, goodbye.")