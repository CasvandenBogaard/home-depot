import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error

df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
df_description = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
df_attributes = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")

N = df_train.shape[0]

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_description, how='left', on='product_uid')

tfv = TfidfVectorizer(ngram_range=(1,2), analyzer='word', stop_words='english')

df_all["product_info"] = df_all['product_title'] + "\t" +  df_all['product_description'] + "\t"+ df_all['search_term']

tfv.fit_transform(df_all["product_info"])
tfidf = tfv.fit_transform(df_all["product_info"])

tsvd = TruncatedSVD(n_components=100, random_state=20)
tfidf_trunc = tsvd.fit_transform(tfidf)

x_train = tfidf_trunc[:N]
x_test = tfidf_trunc[N:]
df_train = df_all.iloc[:N]
y_train = df_train['relevance'].values

clf = RandomForestRegressor(n_estimators = 100)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

id_test = df_all.iloc[N:]['id']

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv',index=False)