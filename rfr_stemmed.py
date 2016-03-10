import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import os.path

def rmse(true, test):
    return mean_squared_error(true, test)**0.5

def features(data):
    df = pd.DataFrame()
    df['descr_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(data['search_term'], data['product_description'])]
    df['title_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(data['search_term'], data['product_title'])]
    df['descr_match'] = [1 if x in y else 0 for x,y in zip(data['search_term'], data['product_description'])]
    df['title_match'] = [1 if x in y else 0 for x,y in zip(data['search_term'], data['product_title'])]
    df['brand_match'] = [1 if str(y) in x else 0 for x,y in zip(data['search_term'], data['brand'])]
    df['attribute_overlap'] = [sum(int(word in str(y)) for word in x.split()) for x,y in zip(data['search_term'], data['joined_attributes'])]
    df['query_length'] = [len(x.split()) for x in data['search_term']]
    
    df = pd.DataFrame({
        'descr_overlap': df['descr_overlap'],
        'title_overlap': df['title_overlap'],
        'descr_match': df['descr_match'],
        'title_match': df['title_match'],
        'brand_match': df['brand_match'],
        'attribute_overlap': df['attribute_overlap'],
        'query_length': df['query_length'],
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
    df_train = pd.read_csv('data/stemmed/train.csv')
    df_description = pd.read_csv('data/stemmed/product_descriptions.csv')
    df_attributes = pd.read_csv('data/stemmed/attributes.csv')
    df_test = pd.read_csv('data/stemmed/test.csv')
else:
    df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
    df_description = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
    df_attributes = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")

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





df_train["product_title"] = df_train["product_title"].map(lambda x:str_stem(x))
df_train["search_term"] = df_train["search_term"].map(lambda x:str_stem(x))
df_train["brand"] = df_train["brand"].map(lambda x:str_stem(x))
df_train["color"] = df_train["color"].map(lambda x:str_stem(x))
df_description["product_description"] = df_description["product_description"].map(lambda x:str_stem(x))


df_train = pd.merge(df_train, df_description, how='left', on='product_uid')

N = df_train.shape[0]

# Cross-validation setup
kf = cross_validation.KFold(N, n_folds=10, random_state=2016)
avg_rmse = 0.
for train, test in kf:
    train_set = df_train.loc[train]
    test_set  = df_train.loc[test]
    x_train, x_test, y_train, y_test = extract_features(train_set, test_set)

    clf = RandomForestRegressor(verbose=True)
    print("Fitting random forest regressor")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

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

extract_sizes(df_train)
x_train, x_test, y_train = extract_features(df_train, df_test)
clf = RandomForestRegressor(n_estimators=30, random_state=26, verbose=True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv', index=False)





