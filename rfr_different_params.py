import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
df_description = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
df_attributes = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")

N = df_train.shape[0]

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_description, how='left', on='product_uid')
df_all['descr_search'] = df_all['product_description'] + "\t" + df_all['search_term']
df_all['descr_overlap'] = df_all['descr_search'].map(lambda x: " ".join(set(x.split("\t")[0].split(" ")).intersection(set(x.split("\t")[1].split(" ")))))
df_all['title_search'] = df_all['product_title'] + "\t" + df_all['search_term']
df_all['title_overlap'] = df_all['title_search'].map(lambda x: " ".join(set(x.split("\t")[0].split(" ")).intersection(set(x.split("\t")[1].split(" ")))))

tfv_descr = TfidfVectorizer(analyzer='word')
tfv_descr.fit(df_all['descr_search'])
td_descr = tfv_descr.transform(df_all['descr_overlap'])
tfv_title = TfidfVectorizer(analyzer='word')
tfv_title.fit(df_all['title_search'])
td_title = tfv_title.transform(df_all['title_overlap'])

tsvd_descr = TruncatedSVD(random_state=30)
result_descr = tsvd_descr.fit_transform(td_descr)
tsvd_title = TruncatedSVD(random_state=30)
result_title = tsvd_title.fit_transform(td_title)

x_train = np.concatenate((result_descr[:N], result_title[:N]), axis=1)
x_test  = np.concatenate((result_descr[N:], result_title[N:]), axis=1)
y = df_all.iloc[:N]['relevance'].values
y_test = df_all.iloc[N:]['relevance'].values
id_test = df_all.iloc[N:]['id']

clf = RandomForestRegressor(n_estimators=20, random_state=30, verbose=True)
clf.fit(x_train, y)

y_pred = clf.predict(x_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv', index=False)