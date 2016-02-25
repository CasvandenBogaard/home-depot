import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
df_description = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
df_attributes = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")

N = df_train.shape[0]

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_description, how='left', on='product_uid')

tfv = TfidfVectorizer(ngram_range=(1,2), analyzer='word')
tfv.fit_transform(df_all['product_title'] + df_all['product_description'] + df_all['search_term']) # takes quite some time, but not too long

df_all['query_length'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['title_length'] = df_all['product_title'].map(lambda x: len(x.split())).astype(np.int64)
df_all['description_length'] = df_all['product_description'].map(lambda x: len(x.split())).astype(np.int64)

df_all_rfr = df_all.drop(['search_term','product_title','product_description'],axis=1)

df_train = df_all_rfr.iloc[:N]
df_test = df_all_rfr.iloc[N:]
id_test = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'relevance'], axis=1).values
X_test = df_test.drop(['id', 'relevance'], axis=1).values

clf = RandomForestRegressor()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv',index=False)