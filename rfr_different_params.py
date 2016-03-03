import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def rmse(true, test):
    return mean_squared_error(true, test)**0.5

def features(data):
    data['descr_overlap'] = [" ".join(set(x.split(" ")).intersection(set(y.split(" ")))) for x,y in zip(data['search_term'], data['product_description'])]
    data['title_overlap'] = [" ".join(set(x.split(" ")).intersection(set(y.split(" ")))) for x,y in zip(data['search_term'], data['product_title'])]
    data['descr_match'] = [1 if x in y else 0 for x,y in zip(data['search_term'], data['product_description'])]
    data['title_match'] = [1 if x in y else 0 for x,y in zip(data['search_term'], data['product_title'])]
    data['brand_match'] = [1 if str(y) in x else 0 for x,y in zip(data['search_term'], data['brand'])]
    data['color_match'] = [1 if str(y) in x else 0 for x,y in zip(data['search_term'], data['color'])]

    df = pd.DataFrame({
        'descr_overlap': data['descr_overlap'].map(lambda x: len(x)),
        'title_overlap': data['title_overlap'].map(lambda x: len(x)),
        'descr_match': data['descr_match'],
        'title_match': data['title_match'],
        'brand_match': data['brand_match'],
        'color_match': data['color_match'],
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

df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
df_description = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
df_attributes = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")

df_train = pd.merge(df_train, df_description, how='left', on='product_uid')

brands = df_attributes[df_attributes.name=='MFG Brand Name']
df_train = pd.merge(df_train, brands, how='left', on='product_uid')
df_train.drop('name',inplace=True,axis=1)
df_train.columns = df_train.columns.str.replace('value','brand')

colors = df_attributes[df_attributes.name=='Color Family']
df_train = pd.merge(df_train, colors, how='left', on='product_uid')
df_train.drop('name',inplace=True,axis=1)
df_train.columns = df_train.columns.str.replace('value','color')

N = df_train.shape[0]

# Cross-validation setup
kf = cross_validation.KFold(N, n_folds=10, random_state=30)
for train, test in kf:
    train_set = df_train.loc[train]
    test_set  = df_train.loc[test]
    x_train, x_test, y_train, y_test = extract_features(train_set, test_set)

    clf = RandomForestRegressor(verbose=True)
    print("Fitting random forest regressor")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(rmse(y_test, y_pred))

# This is for handing in results
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
df_test = pd.merge(df_test, df_description, how='left', on='product_uid')

id_test = df_test['id']

x_train, x_test, y_train = extract_features(df_train, df_test)
clf = RandomForestRegressor(n_estimators=30, random_state=30, verbose=True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv', index=False)