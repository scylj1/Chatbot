'''

Test for Small talk, Question answering

'''
import csv
from main import Information_retrieval, Smalltalk
import json
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore warnings

def smalltalk_test():
    pairs = []
    QAs = csv.reader(open('data/test_dataset.csv','r', errors = 'ignore', encoding = 'utf-8'))
    for qa in QAs:
        pairs.append([qa[0], qa[1]])       
    pairs.pop(0)
    
    st = Smalltalk()
    ir = Information_retrieval()
    accurate = 0
    for pair in pairs:
        similarities = ir.get_similarity_cosine(pair[0], stop = False)
        max_similar = max(similarities)
        if (max_similar < 0.9):         
            response = st.talk_response(pair[0])
            if response != "Sorry, I do not understand":
                if st.predict[0].get('intent') == pair[1]:
                    accurate += 1
    print("Accuracy on small talk test data")
    print(accurate/len(pairs))

def trainingdata_test():
    accurate = 0
    total = 0
    data_file = open('data/smalltalk/intents.json').read()
    intents = json.loads(data_file)
    st = Smalltalk()
    ir = Information_retrieval()
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            similarities = ir.get_similarity_cosine(pattern, stop = False)
            max_similar = max(similarities)
            if (max_similar < 0.9):
                st.talk_response(pattern)
                #print(st.predict)
                if st.predict != []:
                    accurate += 1
            total += 1
    print("Accuracy on training data")
    print(accurate/total)
    
def qa_test():
    pairs = []
    QAs = csv.reader(open('data/QA/QA_dataset.csv','r', errors = 'ignore', encoding = 'utf-8'))
    for qa in QAs:
        pairs.append([qa[1], qa[2]])  
    pairs.pop(0)
    ir = Information_retrieval()
    
    accurate = 0
    for pair in pairs:
        similarities = ir.get_similarity_cosine(pair[0], stop = False)
        max_similar = max(similarities)
        if (max_similar > 0.9):
            max_index = similarities.index(max_similar)
            if ir.pairs[max_index][1] == pair[1]:
                accurate += 1
              
    print("Accuracy on QA data")  
    print(accurate/len(pairs))
    
    
if __name__ == '__main__':
        
    qa_test()
    smalltalk_test()
    trainingdata_test()