import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error

def rmse(true, test):
    return mean_squared_error(true, test)**0.5

def tfidf(data):
    data['descr_search'] = data['product_description'] + "\t" + data['search_term']
    data['descr_overlap'] = data['descr_search'].map(lambda x: " ".join(set(x.split("\t")[0].split(" ")).intersection(set(x.split("\t")[1].split(" ")))))
    data['title_search'] = data['product_title'] + "\t" + data['search_term']
    data['title_overlap'] = data['title_search'].map(lambda x: " ".join(set(x.split("\t")[0].split(" ")).intersection(set(x.split("\t")[1].split(" ")))))

    print("Learning tf idf vectors")
    tfv1 = TfidfVectorizer(analyzer='word')
    tfv1.fit(data['descr_search'])
    td1 = tfv1.transform(data['descr_overlap'])
    tfv2 = TfidfVectorizer(analyzer='word')
    tfv2.fit(data['title_search'])
    td2 = tfv2.transform(data['title_overlap'])

    print("Reducing feature space")
    tsvd1 = TruncatedSVD(random_state=30)
    result1 = tsvd1.fit_transform(td1)
    tsvd2 = TruncatedSVD(random_state=30)
    result2 = tsvd2.fit_transform(td2)
    return np.concatenate((result1,result2), axis=1)

def extract_features(train, test):
    df_all = pd.concat((train, test), axis=0, ignore_index=True)

    result_train = tfidf(train)
    result_test  = tfidf(test)

    x_train = result_train
    x_test  = result_test
    y_train = train['relevance'].values
    if 'relevance' in test:
        y_test = test['relevance'].values
        return x_train, x_test, y_train, y_test
    return x_train, x_test, y_train

df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
df_description = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
df_attributes = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")

df_train = pd.merge(df_train, df_description, how='left', on='product_uid')

N = df_train.shape[0]

kf = cross_validation.KFold(N, n_folds=10, random_state=30)
for train, test in kf:
    train_set = df_train.loc[train]
    test_set  = df_train.loc[test]
    x_train, x_test, y_train, y_test = extract_features(train_set, test_set)

    clf = RandomForestRegressor(n_estimators=20, random_state=30, verbose=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(rmse(y_test, y_pred))


exit(1)

# This is for handing in results
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
df_test = pd.merge(df_test, df_description, how='left', on='product_uid')

id_test = df_test['id']

x_train, x_test, y_train = extract_features(df_train, df_test)
clf = RandomForestRegressor(n_estimators=20, random_state=30, verbose=True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv', index=False)