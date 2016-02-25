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
df_all['product_info'] = df_all['product_title'] + "\t" + df_all['product_description'] + "\t" + df_all['search_term']


tfv = TfidfVectorizer(ngram_range=(1,2), analyzer='word', stop_words='english')
td = tfv.fit_transform(df_all['product_info'])

tsvd = TruncatedSVD(n_components=50)
result = tsvd.fit_transform(td)

x_train = result[:N]
x_test  = result[N:]
y = df_all.iloc[:N]['relevance'].values
id_test = df_all.iloc[N:]['id']

clf = RandomForestRegressor(n_estimators=50)
clf.fit(x_train, y)

y_pred = clf.predict(x_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv',index=False)