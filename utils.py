import os
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import json
import pickle
import random
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()
dirname = os.path.dirname(__file__)


model = load_model(dirname + '/weights/chat_model.h5')
intents = json.loads(open(dirname + '/weights/chat.json').read())
words = pickle.load(open(dirname + '/weights/chat_words.pkl','rb'))
classes = pickle.load(open(dirname + '/weights/chat_classes.pkl','rb'))


class GetResponse:
    
    def __init__(self, msg):
        self.msg = msg
        
    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

    def bow(self, sentence, words, show_details=True):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(words)
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
        return(np.array(bag))

    def predict_class(self, sentence, model):
        # filter out predictions below a threshold
        p = self.bow(sentence, words,show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.2
        results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(self, ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self):
        ints = self.predict_class(self.msg, model)
        if float(ints[0]['probability']) < 0.85:
            return "Could not understand"
        res = self.getResponse(ints, intents)
        return res

    def chatbot_response(self):
        ints = self.predict_class(self.msg, model)
        if float(ints[0]['probability']) < 0.85:
            return "Could not understand"
        if ints[0]['intent'] == 'greeting' and float(ints[0]['probability']) < 0.99:
            return "Could not understand"
        res = self.getResponse(ints, intents)
        return res