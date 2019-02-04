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
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tag.stanford import StanfordPOSTagger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict, KFold

class Preprocessamento(object):
    def __init__(self):
        self.all_tweets = None
        self.polarity_tweets = None
    
    def read_tweets_from_file(self, dataset):
        self.all_tweets = dataset['text'].values
                
        return self.all_tweets
         
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
       
class Resultados(object):
    def __init__(self):
        self.m = Matrix()
        
    def result_unigram(self, tweets_unigram, emoticon, classes):
        matrix1 = self.m.create_matrix(tweets_unigram)
        
        matrix = np.concatenate((matrix1.toarray(), emoticon), axis=1)
        
        #modelo = LinearSVC()
        #modelo = RandomForestClassifier()
        modelo = LogisticRegression()       

        kfold = KFold(n_splits=10, shuffle=True)

        resultados = cross_val_predict(modelo, matrix, classes, cv = kfold)
        
        sentimento = ['positive', 'negative', 'neutral']
                  
        print("Acurácia...: %.2f" %(metrics.accuracy_score(classes,resultados) * 100))
        print("Precision..: %.2f" %(metrics.precision_score(classes,resultados,average='macro') * 100))
        print("Recall.....: %.2f" %(metrics.recall_score(classes,resultados, average='macro') * 100))
        print("F1-Score...: %.2f" %(metrics.f1_score(classes,resultados, average='macro') * 100))
        print(metrics.classification_report(classes,resultados,sentimento,digits=4))
        #print()
        #print(pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')
        
    def result_bigram(self, tweets_bigram, emoticon, classes):
        matrix = self.m.create_matrix(tweets_bigram)
        
        matrix = np.concatenate((matrix.toarray(), emoticon), axis=1)
        
        modelo = LinearSVC()
        #modelo = RandomForestClassifier()
        #modelo = LogisticRegression()
        
        kfold = KFold(n_splits=10, shuffle=True)

        resultados = cross_val_predict(modelo, matrix, classes, cv = kfold)
        
        sentimento = ['positive', 'negative', 'neutral']
                  
        print("Acurácia...: %.2f" %(metrics.accuracy_score(classes,resultados) * 100))
        print("Precision..: %.2f" %(metrics.precision_score(classes,resultados,average='macro') * 100))
        print("Recall.....: %.2f" %(metrics.recall_score(classes,resultados, average='macro') * 100))
        print("F1-Score...: %.2f" %(metrics.f1_score(classes,resultados, average='macro') * 100))
        print(metrics.classification_report(classes,resultados,sentimento,digits=4))
        #print()
        #print(pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')
        
    def result_unigram_e_bigram(self, tweets_unigram, tweets_bigram, emoticon, classes):
        matrix1 = self.m.create_matrix(tweets_unigram)
        matrix2 = self.m.create_matrix(tweets_bigram)
        
        matrix = np.concatenate((matrix1.toarray(), matrix2.toarray()), axis=1)
        matrix = np.concatenate((matrix, emoticon), axis=1)
        
        #modelo = LinearSVC()
        #modelo = RandomForestClassifier()
        modelo = LogisticRegression()
        
        kfold = KFold(n_splits=10, shuffle=True)

        resultados = cross_val_predict(modelo, matrix, classes, cv = kfold)
        
        sentimento = ['positive', 'negative', 'neutral']
                  
        print("Acurácia...: %.2f" %(metrics.accuracy_score(classes,resultados) * 100))
        print("Precision..: %.2f" %(metrics.precision_score(classes,resultados,average='macro') * 100))
        print("Recall.....: %.2f" %(metrics.recall_score(classes,resultados, average='macro') * 100))
        print("F1-Score...: %.2f" %(metrics.f1_score(classes,resultados, average='macro') * 100))
        print(metrics.classification_report(classes,resultados,sentimento,digits=4))
        #print()
        #print(pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')

    
class CriaEmoticon(object):
    def __init__(self):
        self.matriz = []
        
    def cria_emoticon(self, emoji_pos, emoji_neg, emoji_neu, all_tweets):
        for tweet in all_tweets:
            cont = [0 for i in range(3)]

            for word in word_tokenize(tweet.lower()):
                if word in emoji_pos:
                    cont[0] = 1
                elif word in emoji_neg:
                    cont[1] = 1
                else:
                    cont[2] = 1

            self.matriz.append(cont)

        return self.matriz

class LeituraEmoticon(object):
    def __init__(self):
        self.emoji = None
        self.emoji_pos = []
        self.emoji_neg = []
        self.emoji_neu = []

    def leitura(self):
        self.emoji = pd.read_table('Emoticon/EmoticonSentimentLexicon.txt')

        for i in range(len(self.emoji)):
            if list(self.emoji['sentiment'])[i] == 1:
                self.emoji_pos.append(list(self.emoji['emojis'])[i])
            
            elif list(self.emoji['sentiment'])[i] == -1:
                self.emoji_neg.append(list(self.emoji['emojis'])[i])
            
            else:
                self.emoji_neu.append(list(self.emoji['emojis'])[i])

        return self.emoji_pos, self.emoji_neg, self.emoji_neu
    
def main():
    start_ini = time()
    
    dataset = pd.read_csv('sentiment.csv')
    
    pre = Preprocessamento()
    leEmoji = LeituraEmoticon()
    criaEmoji = CriaEmoticon()
    result = Resultados()
    gram = Gram()
    
    ''' Lendo base de dados '''
    
    all_tweets = pre.read_tweets_from_file(dataset)
    classes = pre.read_polarity_from_file(dataset)

    ''' Lendo léxicos positivos e negativos '''

    emoji_pos, emoji_neg, emoji_neu = leEmoji.leitura()

    ''' Cria matrix de emoji '''
    
    matriz_emoji = criaEmoji.cria_emoticon(emoji_pos, emoji_neg, emoji_neu, all_tweets)

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

    
    ''' Classificação dos tweets com Bag of Words + Emoticon '''
    
    #result.result_unigram(tweets_unigram, matriz_emoji, classes)
    #result.result_bigram(tweets_bigram, matriz_emoji, classes)
    result.result_unigram_e_bigram(tweets_unigram, tweets_bigram, matriz_emoji, classes)

    start_fim = time()
    
    tempo = start_fim - start_ini
    h = tempo // 3600
    m = (tempo - h*3600) // 60
    s = (tempo - h*3600) - (m * 60)

    print("\nTempo de execução: %.2i:%.2i:%.2i" %(h,m,s))

main()