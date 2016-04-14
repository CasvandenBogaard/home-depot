from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle

# IMPORTANT: I assume word2vec was already executed so you do have the 'sentences.txt' file

os.chdir('../')

vect = CountVectorizer(input='filename', token_pattern='\\b[^\s]+\\b')
result = vect.fit_transform(['data/word2vec/sentences.txt'])

with open('data/termcounts/vocab.pkl', 'wb') as file:
    pickle.dump(vect.vocabulary_, file)

with open('data/termcounts/counts.pkl', 'wb') as file:
    pickle.dump(result, file)
