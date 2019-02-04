#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#@author: alison

import re
import numpy as np
import pandas as pd
from time import time
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.classify import MaxentClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

TEST_SIZE = None
DATA_SIZE = None

class Preprocessamento(object):
    def __init__(self):
        self.all_tweets = None
        self.polarity_tweets = None
    
    def read_tweets_from_file(self, dataset):
        self.all_tweets = dataset['content']
                
        return self.all_tweets
         
    def read_polarity_from_file(self, dataset):
        self.polarity_tweets = dataset['sentiment']
                       
        return self.polarity_tweets
        
    def clean_tweets(self, tweet):
        tweet = re.sub('@(\w{1,15})\b', '', tweet)
        tweet = tweet.replace("via ", "")
        tweet = tweet.replace("RT ", "")
        tweet = tweet.lower()
        
        return tweet
        
    def clean_url(self, tweet):
        tweet = re.sub(r'(https|http)?://(\w|\.|/|\?|=|&|%)*\b', '', tweet, flags=re.MULTILINE)
        tweet = tweet.replace("http", "")
        tweet = tweet.replace("htt", "")
        
        return tweet
        
    def remove_stop_words(self, tweet):
        english_stops = set(stopwords.words('english'))
        
        tokenizer = RegexpTokenizer("[\w']+")
        
        tweet_token = tokenizer.tokenize(tweet)
            
        words = [w for w in tweet_token if not w in english_stops]
        
        return (" ".join(words))
        
    def stemming_tweets(self, tweet):
        ps = PorterStemmer()

        tweets_stemming = ps.stem(tweet)  
        
        return tweets_stemming
               
class Matrix(object):
    def __init__(self):
        self.matrix = None
        
    def create_matrix(self, tweets):
        count_vect = CountVectorizer(analyzer = "word", binary=True)
        self.matrix = count_vect.fit_transform(tweets)
                
        return self.matrix

class Gram(object):
    def __init__(self):
        self.bigram = None
        self.trigram = None
        
    def create_bigram(self, tweet):
        self.bigram = []
        
        for i in range(len(tweet)-1):
            b_gram = tweet[i] + "_" + tweet[i+1]
            self.bigram.append(b_gram)
            
        return (" ".join(self.bigram))
        
    def create_trigram(self, tweet):
        self.trigram = []
        
        for i in range(len(tweet)-2):
            t_gram = tweet[i] + "_" + tweet[i+1] + "_" + tweet[i+2]
            self.trigram.append(t_gram)
            
        return (" ".join(self.trigram))

class Resultados(object):
    def __init__(self):
        self.m = Matrix()

    def result_unigram(self, tweets_unigram, lex, classes):
        matrix = self.m.create_matrix(tweets_unigram)

        matrix = np.concatenate((matrix.toarray(), lex), axis=1)
        
        #modelo = LinearSVC()
        #modelo = RandomForestClassifier()
        modelo = LogisticRegression()

        size = ((TEST_SIZE * 100) / DATA_SIZE) / 100

        X_train, X_test, y_train, y_test = train_test_split(matrix, classes, test_size=size)

        modelo.fit(X_train, y_train)

        prediction = modelo.predict(X_test)

        sentimento = ['positive', 'negative', 'neutral']

        print("Acurácia...: %.2f" %(metrics.accuracy_score(y_test, prediction) * 100))
        print("Precision..: %.2f" %(metrics.precision_score(y_test, prediction, average='macro') * 100))
        print("Recall.....: %.2f" %(metrics.recall_score(y_test, prediction, average='macro') * 100))
        print("F1-Score...: %.2f" %(metrics.f1_score(y_test, prediction, average='macro') * 100))
        print(metrics.classification_report(y_test, prediction, sentimento,digits=4))
        #print(pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

    
    def result_bigram(self, tweets_bigram, lex, classes):
        matrix = self.m.create_matrix(tweets_bigram)

        matrix = np.concatenate((matrix.toarray(), lex), axis=1)

        modelo = LinearSVC()
        #modelo = RandomForestClassifier()
        #modelo = LogisticRegression()

        size = ((TEST_SIZE * 100) / DATA_SIZE) / 100

        X_train, X_test, y_train, y_test = train_test_split(matrix, classes, test_size=size)
        
        modelo.fit(X_train, y_train)
        
        prediction = modelo.predict(X_test)

        sentimento = ['positive', 'negative', 'neutral']
        
        print("Acurácia...: %.2f" %(metrics.accuracy_score(y_test, prediction) * 100))
        print("Precision..: %.2f" %(metrics.precision_score(y_test, prediction, average='macro') * 100))
        print("Recall.....: %.2f" %(metrics.recall_score(y_test, prediction, average='macro') * 100))
        print("F1-Score...: %.2f" %(metrics.f1_score(y_test, prediction, average='macro') * 100))
        print(metrics.classification_report(y_test, prediction, sentimento,digits=4))
        #print(pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

        
    def result_unigram_e_bigram(self, tweets, lex, classes):
        matrix = self.m.create_matrix(tweets)

        matrix = np.concatenate((matrix.toarray(), lex), axis=1)
        
        #modelo = LinearSVC()
        #modelo = RandomForestClassifier()
        modelo = LogisticRegression()
        
        size = ((TEST_SIZE * 100) / DATA_SIZE) / 100

        X_train, X_test, y_train, y_test = train_test_split(matrix, classes, test_size=size)

        modelo.fit(X_train, y_train)
        
        prediction = modelo.predict(X_test)

        sentimento = ['positive', 'negative', 'neutral']
        
        print("Acurácia...: %.2f" %(metrics.accuracy_score(y_test, prediction) * 100))
        print("Precision..: %.2f" %(metrics.precision_score(y_test, prediction, average='macro') * 100))
        print("Recall.....: %.2f" %(metrics.recall_score(y_test, prediction, average='macro') * 100))
        print("F1-Score...: %.2f" %(metrics.f1_score(y_test, prediction, average='macro') * 100))
        print(metrics.classification_report(y_test, prediction, sentimento,digits=4))
        #print(pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

class CriaLexicon(object):
    def __init__(self):
        self.matriz = []
        
    def cria_lexicon(self, lex_positivo, lex_negativo, all_tweets):
        for tweet in all_tweets:
            cont = [0 for i in range(3)]
            contPos = 0
            contNeg = 0

            for word in word_tokenize(tweet.lower()):
                if word in lex_positivo:
                    contPos += 1

                if word in lex_negativo:
                    contNeg += 1

            if contPos > contNeg:
                cont[0] = 1
            elif contNeg > contPos:
                cont[1] = 1
            else:
                cont[2] = 1

            self.matriz.append(cont)

        return self.matriz

class LeituraLexicon(object):
    def __init__(self):
        self.pos = None
        self.neg = None

    def leitura_pos(self):
        self.pos = pd.read_csv('OpinionLexicon/positive-words.csv')

        return list(self.pos['lexicon_pos'])

    def leitura_neg(self):
        self.neg = pd.read_csv('OpinionLexicon/negative-words.csv')

        return list(self.neg['lexicon_neg'])

class RetiraPolaridade(object):
    def __init__(self):
        self.all_tweets = None
        self.polaridade = None

    def retira_polaridade(self, tweets, polaridade):
        self.all_tweets = []
        self.polaridade = []

        for i in range(len(tweets)):
            if polaridade[i] == 'positive' or polaridade[i] == 'negative' or polaridade[i] == 'neutral':
                self.all_tweets.append(tweets[i])
                self.polaridade.append(polaridade[i])

        return self.all_tweets, self.polaridade

def main():
    start_ini = time()
    
    global TEST_SIZE
    global DATA_SIZE
   
    dataset_train = pd.read_csv('hcr-train.csv')
    dataset_test = pd.read_csv('hcr-test.csv')
    
    rp = RetiraPolaridade()
    pre = Preprocessamento()
    leLex = LeituraLexicon()
    criaLex = CriaLexicon()
    result = Resultados()
    gram = Gram()
    
    ''' Lendo base de dados '''
    
    tweets_train = pre.read_tweets_from_file(dataset_train)
    polarity_train = pre.read_polarity_from_file(dataset_train)

    tweets_test = pre.read_tweets_from_file(dataset_test)
    polarity_test = pre.read_polarity_from_file(dataset_test)

    tweets_train, polarity_train = rp.retira_polaridade(tweets_train, polarity_train)
    tweets_test, polarity_test = rp.retira_polaridade(tweets_test, polarity_test)

    TEST_SIZE = len(tweets_test)

    ''' Mesclando os dados de treino com os dados de teste '''
    
    all_tweets = []
    classes = []
    
    for tweet in tweets_train:
    	all_tweets.append(tweet)
    	    
    for tweet in tweets_test:
    	all_tweets.append(tweet)
    	    
    for classe in polarity_train:
        classes.append(classe)
    	    
    for classe in polarity_test:
    	classes.append(classe)

    DATA_SIZE = len(all_tweets)

    ''' Lendo léxicos positivos e negativos '''

    lex_positivo = leLex.leitura_pos()
    lex_negativo = leLex.leitura_neg()

    ''' Cria matrix de léxico '''
    
    matriz_lex = criaLex.cria_lexicon(lex_positivo, lex_negativo, all_tweets)

    ''' Preprocessamento '''

    for i in range(len(all_tweets)):
        all_tweets[i] = pre.clean_tweets(all_tweets[i])
        all_tweets[i] = pre.clean_url(all_tweets[i])
        all_tweets[i] = pre.remove_stop_words(all_tweets[i])
        all_tweets[i] = pre.stemming_tweets(all_tweets[i])
	    
        
    ''' Gerando n-gram '''

    tweets_unigram = all_tweets
    tweets_bigram = []

    for i in range(len(all_tweets)):
        tweets_bigram.append(gram.create_bigram(all_tweets[i].split()))
	        
    ''' Classificação dos tweets no modelo Bag of Words '''

    unigram_and_bigram = []

    for i in range(len(tweets_unigram)):
        unigram_and_bigram.append(tweets_unigram[i] + tweets_bigram[i])

    #result.result_unigram(tweets_unigram, matriz_lex, classes)
    #result.result_bigram(tweets_bigram, matriz_lex, classes)
    result.result_unigram_e_bigram(unigram_and_bigram, matriz_lex, classes)

    start_fim = time()

    tempo = start_fim - start_ini

    h = tempo // 3600
    m = (tempo - h*3600) // 60
    s = (tempo - h*3600) - (m * 60)

    print("\nTempo de execução: %.2i:%.2i:%.2i" %(h,m,s))

main()
