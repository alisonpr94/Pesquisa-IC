#-*- encoding: utf-8 -*-
#@author: alison

import re
import sys
import numpy as np
import pandas as pd
from time import time
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict, KFold

class Preprocessamento(object):
	def __init__(self):
		self.all_twitter_messages = None
		self.polarity_tweets = None
		self.tweets_stemming = None
		self.palavras = []

	def read_tweets_from_file(self, dataset):
		self.all_twitter_messages = dataset['text'].values

		return self.all_twitter_messages

	def read_polarity_from_file(self, dataset):
		self.polarity_tweets = dataset['class'].values

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

		words = [i for i in tweet.split() if not i in english_stops]

		return (" ".join(words))

	def stemming_tweets(self, tweet):
		ps = PorterStemmer()

		self.tweets_stemming = ps.stem(tweet)

		return self.tweets_stemming

class Classification(object):
	def __init__(self):
		self.svc = LinearSVC(C=10.0)
		self.rf = RandomForestClassifier()
		self.lr = LogisticRegression(C=10.0)
		self.modelo = {'svc': self.svc, 'rf': self.rf, 'lr': self.lr}

	def classifying(self, matrix_embedding, classes):	
		model = self.modelo['lr']

		kfold = KFold(n_splits=10, shuffle=True)

		resultados = cross_val_predict(model, matrix_embedding, classes, cv = kfold)
		
		sentimento = ['positive', 'negative', 'neutral']

		print("Acurácia...: %.2f" %(metrics.accuracy_score(classes,resultados) * 100))
		print("Precision..: %.2f" %(metrics.precision_score(classes,resultados,average='macro') * 100))
		print("Recall.....: %.2f" %(metrics.recall_score(classes,resultados, average='macro') * 100))
		print("F1-Score...: %.2f" %(metrics.f1_score(classes,resultados, average='macro') * 100))
		#print()
		print(metrics.classification_report(classes,resultados,sentimento,digits=4))

	def ensemble(self, matrix_embedding, classes):
		voting = VotingClassifier(estimators=[('svc', self.svc), ('rf', self.rf), ('lr', self.lr)], voting='hard')

		kfold = KFold(n_splits=10, shuffle=True)

		resultados = cross_val_predict(voting, matrix_embedding, classes, cv = kfold)
		
		sentimento = ['positive', 'negative', 'neutral']

		print("Acurácia...: %.2f" %(metrics.accuracy_score(classes,resultados) * 100))
		print("Precision..: %.2f" %(metrics.precision_score(classes,resultados,average='macro') * 100))
		print("Recall.....: %.2f" %(metrics.recall_score(classes,resultados, average='macro') * 100))
		print("F1-Score...: %.2f" %(metrics.f1_score(classes,resultados, average='macro') * 100))
		#print()
		print(metrics.classification_report(classes,resultados,sentimento,digits=4))

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class WordEmbeddings(object):
	def __init__(self):
		self.w2v = None
		self.matrix = None

	def glove_reading(self, path_dimension, all_tweets):
		if path_dimension == 50:
			with open("glove.6B.50d.txt", "rb") as lines:
				self.w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
					for line in lines}
            
		elif path_dimension == 100:
			with open("glove.6B.100d.txt", "rb") as lines:
				self.w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
					for line in lines}
		
		elif path_dimension == 200:
			with open("glove.6B.200d.txt", "rb") as lines:
				self.w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
					for line in lines}

		elif path_dimension == 300:
			with open("glove.6B.300d.txt", "rb") as lines:
				self.w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
					for line in lines}

		else:
			print("Escolha da dimensão incorreta!")

		vec = MeanEmbeddingVectorizer(self.w2v)
		vec.fit(all_tweets)
		self.matrix = vec.transform(all_tweets)

		return self.matrix

def bag_of_words(tweets):
	vec = CountVectorizer(analyzer="words")
	vec.fit(tweets)
	matrix = vec.transform(tweets).toarray()

	return matrix

def opinion_lexicon(all_tweets):
	lex_positivo = pd.read_csv('opinion_lexicon/positive-words.csv')
	lex_negativo = pd.read_csv('opinion_lexicon/negative-words.csv')

	matrix = []

	for tweet in all_tweets:
		vec = np.zeros(3)
		contPos = 0
		contNeg = 0

		for word in word_tokenize(tweet.lower()):
			if word in list(lex_positivo):
				contPos += 1

			if word in list(lex_negativo):
				contNeg += 1

		if contPos > contNeg:
			vec[0] = 1.0			#Negativo
		elif contNeg > contPos:
			vec[1] = 1.0			#Positivo
		else:
			vec[2] = 1.0			#Neutro

		matrix.append(vec)

	return matrix

def main(path_dimension):
	start_ini = time()

	dataset = pd.read_csv('sentiment.csv')

	''' Instâncias '''

	pre = Preprocessamento()
	we = WordEmbeddings()
	classifier = Classification()

	''' Leitura dataset '''

	tweets = pre.read_tweets_from_file(dataset)
	classes = pre.read_polarity_from_file(dataset)

	matrix_lex = opinion_lexicon(tweets)

	''' Preprocessamento '''

	for i in range(len(tweets)):
		tweets[i] = pre.clean_tweets(tweets[i])
		tweets[i] = pre.clean_url(tweets[i])
		tweets[i] = pre.remove_stop_words(tweets[i])

	matrix_embedding = we.glove_reading(path_dimension, tweets)
	matrix_bow = bag_of_words(tweets)

	matrix = np.concatenate((matrix_embedding, matrix_lex, matrix_bow), axis=1)
	
	classifier.classifying(matrix, classes)
	#classifier.ensemble(matrix_embedding, classes)

	start_fim = time()

	tempo = start_fim - start_ini
	h = tempo // 3600
	m = (tempo - h*3600) // 60
	s = (tempo - h*3600) - (m * 60)

	print("\nTempo de execução: %.2i:%.2i:%.2i" %(h,m,s))

if __name__ == '__main__':
	args = sys.argv[1:]

	if len(args) == 1:

		path_dimension = args[0]
		main(int(path_dimension))

	else:
		sys.exit('''
			Requires:
			path_embedding -> Path of the word embedding
			path_algorithm -> Path of the algorithm
			''')
