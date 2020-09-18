import csv
import string

import nltk
from nltk.corpus import nps_chat
import random

documents = []
all_chat_words = []
with open('raw_data/train.csv','r') as train_file:
    reader = csv.reader(train_file)
    for row in reader:
        #change 1 to insult, 2 to not_insult
        classification = row[0]
        if classification == '0':
            classification = 'not_insult'
        elif classification == '1':
            classification = 'insult'
        #extract chat from csv file
        chat = row[2]
        #removes unnecesary quote marks
        chat = chat.replace('"', '')
        tokens = nltk.word_tokenize(chat)
        pair = [tokens, classification]
        documents.append(pair)
        all_chat_words.extend(tokens)
random.shuffle(documents)

#gets most commonly used words
all_words = nltk.FreqDist(all_chat_words)
word_features = list(all_words)[:2000]

#assigns boolean values for each of most commonly used words
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

#data-set length is 3947
featuresets = [(document_features(d), c) for (d, c) in documents]
train_set = featuresets[3000:]
devtest_set = featuresets[1000:3000]
train_set = featuresets[:947]
#train classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))
classifier.show_most_informative_features(20)