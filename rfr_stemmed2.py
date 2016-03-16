import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer

stemmer = PorterStemmer()
#stemmer = SnowballStemmer('english')

def rmse(true, test):
    return mean_squared_error(true, test)**0.5

def features(data):
    df = pd.DataFrame()
    df['descr_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(data['search_term'], data['product_description'])]
    df['title_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(data['search_term'], data['product_title'])]
    df['descr_match'] = [1 if x in y else 0 for x,y in zip(data['search_term'], data['product_description'])]
    df['title_match'] = [1 if x in y else 0 for x,y in zip(data['search_term'], data['product_title'])]
    df['brand_match'] = [1 if str(y) in x else 0 for x,y in zip(data['search_term'], data['brand'])]
    #df['color_match'] = [1 if str(y) in x else 0 for x,y in zip(data['search_term'], data['color'])]
    #df['query_length'] = data['search_term'].map(lambda x:len(x.split()))
    df['query_length'] = [len(x.split()) for x in data['search_term']]
    df['attribute_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(data['search_term'], data['joined_attributes'])]
    
    df = pd.DataFrame({
        'descr_overlap': df['descr_overlap'],
        'title_overlap': df['title_overlap'],
        'descr_match': df['descr_match'],
        'title_match': df['title_match'],
        'brand_match': df['brand_match'],
        #'color_match': df['color_match'],
        'query_length': df['query_length'],
        'attribute_overlap': df['attribute_overlap']
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

def extract_sizes(data):
    pattern = '(\d+(\.\d+)? xbi \d+(\.*\d+)?)( xbi \d+(\.\d+)?)*'
    product_lengths = ' '.join([re.search(pattern, search).group(0).split(" xbi ") if re.search(pattern,search) else "none" for search in data['search_term']])

def str_stem(s): 
    if isinstance(s, str):
        s = s.lower()
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) ##'desgruda' palavras que estão juntas

        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
    
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
    
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
    
        s = s.replace(" x "," xby ")
        s = s.replace("*"," xby ")
        s = s.replace(" by "," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
        s = s.replace("0x","0 xby ")
        s = s.replace("1x","1 xby ")
        s = s.replace("2x","2 xby ")
        s = s.replace("3x","3 xby ")
        s = s.replace("4x","4 xby ")
        s = s.replace("5x","5 xby ")
        s = s.replace("6x","6 xby ")
        s = s.replace("7x","7 xby ")
        s = s.replace("8x","8 xby ")
        s = s.replace("9x","9 xby ")
        
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
    
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
    
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
    
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
    
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
    
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        
        s = s.replace("&amp;", "and")
        s = s.replace("&", "and")        
        
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")

        s = s.replace("  "," ")
        s = (" ").join([stemmer.stem(z) for z in s.lower().split(" ")])
        #s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        return s.lower()
    else:
        return "null"

def norm_length(s):
    pattern = '.*(Width|Height|Depth|Length).*'
    if(re.match(pattern, s)):
        return "Length_Norm"
    else:
        return s

df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1").head(50)
df_description = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1").head(1000)
df_attributes = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1").head(1000)



brands = df_attributes[df_attributes.name=='MFG Brand Name']
df_train = pd.merge(df_train, brands, how='left', on='product_uid')
df_train.drop('name',inplace=True,axis=1)
df_train.columns = df_train.columns.str.replace('value','brand')

colors = df_attributes[df_attributes.name=='Color Family']
df_train = pd.merge(df_train, colors, how='left', on='product_uid')
df_train.drop('name',inplace=True,axis=1)
df_train.columns = df_train.columns.str.replace('value','color')


df_train["product_title"] = df_train["product_title"].map(lambda x:str_stem(x))
df_train["search_term"] = df_train["search_term"].map(lambda x:str_stem(x))
df_train["brand"] = df_train["brand"].map(lambda x:str_stem(x))
df_train["color"] = df_train["color"].map(lambda x:str_stem(x))
df_description["product_description"] = df_description["product_description"].map(lambda x:str_stem(x))
df_attributes["value"] = df_attributes["value"].map(lambda x:str_stem(x))


df_train = pd.merge(df_train, df_description, how='left', on='product_uid')

N = df_train.shape[0]


df_attributes = df_attributes.drop('name', axis=1)
df_attributes["value"] = df_attributes["value"].astype(str)



grouped = df_attributes.groupby('product_uid')
df_train = pd.merge(df_train, grouped.apply(lambda x: (" ").join(x.value)), how='left', on='product_uid' )


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

    print(clf.feature_importances_)

    
    rmse_fold = rmse(y_test, y_pred)
    print(rmse_fold)
    avg_rmse += rmse_fold
avg_rmse = avg_rmse/10
print("Avg rmse: " + str(avg_rmse))
print(x_train)


grouped.apply(lambda x: (" ").join(x.value))



df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
df_test = pd.merge(df_test, df_description, how='left', on='product_uid')

brands = df_attributes[df_attributes.name=='MFG Brand Name']
df_test = pd.merge(df_test, brands, how='left', on='product_uid')
df_test.drop('name',inplace=True,axis=1)
df_test.columns = df_test.columns.str.replace('value','brand')

#colors = df_attributes[df_attributes.name=='Color Family']
#df_test = pd.merge(df_test, colors, how='left', on='product_uid')
#df_test.drop('name',inplace=True,axis=1)
#df_test.columns = df_test.columns.str.replace('value','color')



df_test["product_title"] = df_test["product_title"].map(lambda x:str_stem(x))
df_test["search_term"] = df_test["search_term"].map(lambda x:str_stem(x))



id_test = df_test['id']

x_train, x_test, y_train = extract_features(df_train, df_test)
clf = RandomForestRegressor(n_estimators=30, random_state=26, verbose=True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv', index=False)



x_train['brand']





