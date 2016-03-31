import pandas as pd
import os
from gensim.models.word2vec import *

os.chdir('../')
# use stemming.py if you want to stem
if (os.path.isdir('data/stemmed')):
    df_train = pd.read_csv('data/stemmed/train.csv', encoding="ISO-8859-1")
    df_description = pd.read_csv('data/stemmed/product_descriptions.csv', encoding="ISO-8859-1")
    df_attributes = pd.read_csv('data/stemmed/attributes.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('data/stemmed/test.csv', encoding="ISO-8859-1")
else:
    df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
    df_description = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
    df_attributes = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")

if not os.path.isfile('data/word2vec/sentences.txt'):
    with open('data/word2vec/sentences.txt', 'w') as file:
        sentences = []

        sentences = [x for x in df_train['search_term']]
        sentences = sentences + [x for x in df_train['product_title']]
        sentences = sentences + [x for x in df_test['search_term']]
        sentences = sentences + [x for x in df_test['product_title']]
        sentences = sentences + [x for x in df_attributes['value']]
        descr_sentences = [x.split(".") for x in df_description['product_description']]
        sentences = sentences + [x for sublist in descr_sentences for x in sublist]

        sentencegen = (x for x in sentences if x != "")
        for sentence in sentencegen:
            file.write("{}\n".format(sentence.lstrip()))

sentences = LineSentence('data/word2vec/sentences.txt')


print("Starting with training of word2vec")
model = Word2Vec(sentences, min_count=5, workers=8)

print("Done! Saving result in data/word2vec")
model.save('data/word2vec/full')

print("Testing similarity between 'toilet' and 'seat':")
print(model.similarity('toilet', 'seat'))