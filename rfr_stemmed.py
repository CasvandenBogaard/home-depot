import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import os.path
from decimal import Decimal
import matplotlib.pyplot as plt
import nltk

def rmse(true, test):
    return mean_squared_error(true, test)**0.5

def n_gram(attribute, data, n_gram_attr, n):
    data[str(n_gram_attr)] = data[str(attribute)]

    ngrams = lambda b, n: [b[i:i+n] for i in range(len(b)-n+1)]
    wordlist = [x.split() for x in data[str(attribute)]]

    ng = [[ngrams(y,n) if len(y) >= n else [y] for y in x] for x in wordlist]
    result = [[item for sublist in x for item in sublist] for x in ng]
    data[str(n_gram_attr)] = [" ".join(x) for x in result]

def find_num_nouns(attribute, data, post_attr):
    sentencelist = [x.split() for x in data[str(attribute)]]

    print(len(sentencelist))
    result = nltk.pos_tag_sents(sentencelist)
    nouns = [[word for word,pos in lst if pos not in ['NN', 'NNP', 'NNS', 'NNPS']] for lst in result]
    data[str(post_attr)] = [len(x) for x in nouns]
    print(len(data[str(post_attr)]))


def features(data):
    df = pd.DataFrame()
    df['descr_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(data['search_term'], data['product_description'])]
    df['descr_overlap_jc'] = [z / (len(x.split()) + len(y.split()) - z)  for x,y,z in zip(data['search_term'], data['product_description'], df['descr_overlap'])]
    df['title_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(data['search_term_ngram'], data['product_title_ngram'])]
    df['title_overlap_jc'] = [z / (len(x.split()) + len(y.split()) - z)  for x,y,z in zip(data['search_term_ngram'], data['product_title_ngram'], df['title_overlap'])]
    df['descr_match'] = [1 if x in y else 0 for x,y in zip(data['search_term'], data['product_description'])]
    df['title_match'] = [1 if x in y else 0 for x,y in zip(data['search_term_ngram'], data['product_title_ngram'])]
    df['brand_match'] = [1 if str(y) in x else 0 for x,y in zip(data['search_term'], data['brand'])]
    df['color_match'] = [1 if str(y) in x else 0 for x,y in zip(data['search_term'], data['joined_attributes'])]
    df['attribute_overlap'] = [sum(int(word in str(y)) for word in x.split()) for x,y in zip(data['search_term'], data['joined_attributes'])]
    df['query_length'] = [len(x.split()) for x in data['search_term_ngram']]
    df['total_match_title'] = [math.floor(x/y) for x,y in zip(df['title_overlap'], df['query_length'])]
    df['query_char_length'] = [len(x) for x in data['search_term']]
    df['query_avg_length'] = [y/x for x,y in zip(df['query_length'], df['query_char_length'])]
    df['numbers'] = [sum(s.isdigit() for s in x.split()) for x in data['search_term']]
    df['number_st_rel'] = [y/x for x,y in zip(df['query_char_length'], df['numbers'])]
    df['num_nouns'] = [int(x) for x in data['num_nouns']]
    
    df = pd.DataFrame({
        'descr_overlap': df['descr_overlap'],
        'descr_overlap_jc': df['descr_overlap_jc'],
        'title_overlap': df['title_overlap'],
        'title_overlap_jc': df['title_overlap_jc'],
        'descr_match': df['descr_match'],
        'title_match': df['title_match'],
        'brand_match': df['brand_match'],
        'color_match': df['color_match'],
        'attribute_overlap': df['attribute_overlap'],
        'query_length': df['query_length'],
        'query_char_length': df['query_char_length'],
        'total_match_title': df['total_match_title'],
        'number_st_rel': df['number_st_rel'],
        'num_nouns': df['num_nouns']
    })

    return df.iloc[:]

def extract_features(train, test):
    result_train = features(train)
    result_test  = features(test)

    x_train = result_train
    x_test  = result_test
    y_train = train['relevance'].values
    if 'relevance' in test:
        y_test = test['relevance'].values
        return x_train, x_test, y_train, y_test
    return x_train, x_test, y_train

# use preprocessing.py if you want to stem
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


n_gram('search_term', df_train, 'search_term_ngram', 3)
n_gram('product_title', df_train, 'product_title_ngram', 3)
n_gram('search_term', df_test, 'search_term_ngram', 3)
n_gram('product_title', df_test, 'product_title_ngram', 3)

find_num_nouns('search_term', df_train, 'num_nouns')
find_num_nouns('search_term', df_test, 'num_nouns')

brands = df_attributes[df_attributes.name=='MFG Brand Name']
df_train = pd.merge(df_train, brands, how='left', on='product_uid')
df_train.drop('name',inplace=True,axis=1)
df_train.columns = df_train.columns.str.replace('value','brand')


color_attribute_mask = ['Color' in str(name) for name in df_attributes['name']]
colors = df_attributes.loc[color_attribute_mask,:]
df_atrr = colors.drop('name', axis=1)
df_atrr["value"] = df_atrr["value"].astype(str)
grouped = df_atrr.groupby('product_uid').apply(lambda x: (" ").join(x.value))
groupeddf = grouped.reset_index()
groupeddf.columns = ['product_uid', 'joined_attributes']
df_train = pd.merge(df_train, groupeddf, how='left', on='product_uid')

df_train = pd.merge(df_train, df_description, how='left', on='product_uid')


N = df_train.shape[0]

# Cross-validation setup
kf = cross_validation.KFold(N, n_folds=10, shuffle=True)
avg_rmse = 0.

for train, test in kf:
    train_set = df_train.loc[train]
    test_set  = df_train.loc[test]
    x_train, x_test, y_train, y_test = extract_features(train_set, test_set)

    clf = RandomForestRegressor(n_estimators=100, max_depth=11, n_jobs=-1)
    print("Fitting random forest regressor")
    
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    names = list(x_train.columns.values)

    uniques = np.unique(y_test)

    # results = []
    # for u in uniques:
    #     indexes = np.where(y_test==u)
    #     results.append(y_pred[indexes])
    #
    # plt.boxplot(results)
    # plt.plot(range(1, len(uniques) + 1), uniques, 'o')
    # plt.xticks(range(1, len(uniques) + 1), uniques)
    # plt.ylim((0,4))
    # plt.show()
    #
    # print(x_test.iloc[0])
    # print(clf.feature_importances_)

    rmse_fold = rmse(y_test, y_pred)
    print(rmse_fold)
    avg_rmse += rmse_fold

avg_rmse = avg_rmse/10
print("Avg rmse: " + str(avg_rmse))


df_test = pd.merge(df_test, df_description, how='left', on='product_uid')
df_test = pd.merge(df_test, brands, how='left', on='product_uid')
df_test.drop('name',inplace=True,axis=1)
df_test.columns = df_test.columns.str.replace('value','brand')

df_test = pd.merge(df_test, groupeddf, how='left', on='product_uid')

id_test = df_test['id']

x_train, x_test, y_train = extract_features(df_train, df_test)
clf = RandomForestRegressor(n_estimators=100, max_depth=11, n_jobs=-1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv', index=False)