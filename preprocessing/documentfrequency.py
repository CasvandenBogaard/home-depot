from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter

# IMPORTANT: I assume word2vec was already executed so you do have the 'sentences.txt' file

os.chdir('../')



DO_THIS = ["df"]

if "tf" in DO_THIS:
    vect = CountVectorizer(input='filename', token_pattern='[^\\s]+')
    result = vect.fit_transform(['data/word2vec/sentences.txt'])

    dict = {}
    for word in vect.vocabulary_:
        dict[word] = result[0, vect.vocabulary_.get(word)]

    with open('data/termcounts/dict.pkl', 'wb') as file:
        pickle.dump(dict, file)

    with open('data/termcounts/vocab.pkl', 'wb') as file:
        pickle.dump(vect.vocabulary_, file)

    with open('data/termcounts/counts.pkl', 'wb') as file:
        pickle.dump(result, file)

if "df" in DO_THIS:
    df_train = pd.read_csv('data/stemmed/train.csv', encoding="ISO-8859-1")
    df_description = pd.read_csv('data/stemmed/product_descriptions.csv', encoding="ISO-8859-1")
    df_attributes = pd.read_csv('data/stemmed/attributes.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('data/stemmed/test.csv', encoding="ISO-8859-1")

    queries = np.concatenate([df_train['search_term'], df_test['search_term']])
    print("Query shape: {}".format(queries.shape))

    # calculate words frequencies per document
    word_frequencies = [Counter(document.split()) for document in queries]

    # calculate document frequency
    doc_freqs = Counter()
    [doc_freqs.update(word_frequency.keys()) for word_frequency in word_frequencies]

    with open('data/termcounts/doccounts.pkl', 'wb') as file:
        pickle.dump(doc_freqs, file)

    titles = np.concatenate([df_train['product_title'], df_test['product_title']])

    titles = np.unique(titles)

    prod_frequencies = [Counter(document.split()) for document in titles]
    prod_freqs = Counter()
    [prod_freqs.update(word_frequency.keys()) for word_frequency in prod_frequencies]

    with open('data/termcounts/prodcounts.pkl', 'wb') as file:
        pickle.dump(prod_freqs, file)