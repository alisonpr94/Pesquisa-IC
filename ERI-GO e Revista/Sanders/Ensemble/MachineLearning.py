#!/usr/bin/env python3

#@author: alison

import re
import numpy as np
import pandas as pd
from time import time
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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

class Gram(object):
	def __init__(self):
		self.bigram = None
		self.trigram = None

	def create_bigram(self, tweet):
		self.bigram  = []

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

class Matrix(object):
	def __init__(self):
		self.matrix = None


	def create_matrix(self, tweets):
		count_vect = CountVectorizer(analyzer = "word", binary=True)
		self.matrix = count_vect.fit_transform(tweets)

		return self.matrix

class Classification(object):
	def __init__(self):
		self.m = Matrix()

	def ensemble(self, all_tweets, classes):
		matrix = self.m.create_matrix(all_tweets)

		#svc = SVC(kernel='linear', probability=True)
		svc = LinearSVC()
		rf = RandomForestClassifier()
		lr = LogisticRegression()
		voting = VotingClassifier(estimators=[('svc', svc), ('rf', rf), ('lr', lr)], voting='hard')
		
		kfold = KFold(n_splits=10, shuffle=True)
		
		resultados = cross_val_predict(voting, matrix, classes, cv=kfold)

		sentimento = ['positive', 'negative', 'neutral']

		print("Acurácia...: %.2f" %(metrics.accuracy_score(classes,resultados) * 100))
		print("Precision..: %.2f" %(metrics.precision_score(classes,resultados,average='macro') * 100))
		print("Recall.....: %.2f" %(metrics.recall_score(classes,resultados, average='macro') * 100))
		print("F1-Score...: %.2f" %(metrics.f1_score(classes,resultados, average='macro') * 100))
		print(metrics.classification_report(classes,resultados,sentimento,digits=4))
		#print()
		#print(pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')

def main():
	start_ini = time()

	dataset = pd.read_csv('sentiment.csv')

	''' Instâncias '''

	pre = Preprocessamento()
	classifier = Classification()
	gram = Gram()

	''' Leitura dataset '''

	tweets = pre.read_tweets_from_file(dataset)
	classes = pre.read_polarity_from_file(dataset)

	''' Preprocessamento '''

	for i in range(len(tweets)):
		tweets[i] = pre.clean_tweets(tweets[i])
		tweets[i] = pre.clean_url(tweets[i])
		tweets[i] = pre.remove_stop_words(tweets[i])
		tweets[i] = pre.stemming_tweets(tweets[i])


	''' Contrução de n-gram '''

	tweets_unigram = tweets
	tweets_bigram = []

	for i in range(len(tweets)):
		tweets_bigram.append(gram.create_bigram(tweets[i].split()))

	tweets_unigram_bigram = tweets_unigram + tweets_bigram

	classifier.ensemble(tweets_unigram_bigram, classes)

	start_fim = time()

	tempo = start_fim - start_ini
	h = tempo // 3600
	m = (tempo - h*3600) // 60
	s = (tempo - h*3600) - (m * 60)

	#print("\nTempo de execução: {}".format(tempo))
	print("\nTempo de execução: %.2i:%.2i:%.2i" %(h,m,s))

main()
