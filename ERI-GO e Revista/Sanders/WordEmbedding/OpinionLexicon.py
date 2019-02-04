import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

class CriaLexicon(object):
	def __init__(self):
		self.matriz = []
        
	def opinion_lexicon(self, lex_positivo, lex_negativo, all_tweets):
		for tweet in all_tweets:
			cont = [0.0 for i in range(3)]
			contPos = 0
			contNeg = 0

			for word in word_tokenize(tweet.lower()):
				if word in lex_positivo:
					print('pos')
					contPos += 1

				if word in lex_negativo:
					print('neg')
					contNeg += 1

			#print(contPos, contNeg)
			if contPos > contNeg:
				cont[0] = 1.0
			elif contNeg > contPos:
				cont[1] = 1.0
			else:
				cont[2] = 1.0

			self.matriz.append(cont)

		return self.matriz

if __name__ == '__main__':
	lex = CriaLexicon()

	dataset = pd.read_csv('sentiment.csv')

	pos = pd.read_csv('opinion_lexicon/positive-words.csv')
	neg = pd.read_csv('opinion_lexicon/negative-words.csv')

	pos = pos['pos']
	neg = neg['neg']

	matrix = lex.opinion_lexicon(list(pos), list(neg), dataset['text'])

	np.savetxt('lex.txt', matrix, fmt="%.5f")
