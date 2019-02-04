from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec

'''
def reading_glove(tweets, dim):
    if dim == 25:
        with open("glove.twitter.27B.25d.txt", "rb") as lines:
            glove = {line.split()[0]: np.array(map(float, line.split()[1:]))
                for line in lines}

    elif dim == 50:
        with open("glove.twitter.27B.50d.txt", "rb") as lines:
            glove = {line.split()[0]: np.array(map(float, line.split()[1:]))
                for line in lines}

    elif dim == 100:
        with open("glove.twitter.27B.100d.txt", "rb") as lines:
            glove = {line.split()[0]: np.array(map(float, line.split()[1:]))
                for line in lines}

    elif dim == 200:
        with open("glove.twitter.27B.200d.txt", "rb") as lines:
            glove = {line.split()[0]: np.array(map(float, line.split()[1:]))
                for line in lines}

    else:
        raise IOError("Dimensão do Word Embedding GloVe incorreta.")
'''
modelo = 'glove.twitter.27B.25d.txt'
model = Word2Vec.load(modelo)
