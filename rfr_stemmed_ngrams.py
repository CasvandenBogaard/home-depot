import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.neural_network import MLPRegressor
# gradient boosting regressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import os.path
import random
random.seed(2016)
from nltk.util import ngrams
import math 

def rmse(true, test):
    return mean_squared_error(true, test)**0.5

def features(data):
    #n = 3
    df = pd.DataFrame()
    df['descr_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(data['search_term'], data['product_description'])]
    df['title_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(data['search_term'], data['product_title'])]
    #df['title_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(data['search_term_ngram'], data['product_title_ngram'])]
    df['descr_match'] = [1 if x in y else 0 for x,y in zip(data['search_term'], data['product_description'])]
    df['title_match'] = [1 if x in y else 0 for x,y in zip(data['search_term'], data['product_title'])]
    df['brand_match'] = [1 if str(y) in x else 0 for x,y in zip(data['search_term'], data['brand'])]
    df['attribute_overlap'] = [sum(int(word in str(y)) for word in x.split()) for x,y in zip(data['search_term'], data['joined_attributes'])]
    #df['attribute_match'] = [1 if x in y else 0 for x,y in zip(data['search_term'], data['joined_attributes'])]
    df['query_length'] = [len(x.split()) for x in data['search_term']]
    #df['query_length'] = [len(x.split()) for x in data['search_term_ngram']]
    df['total_match_title'] = [math.floor(x/y) for x,y in zip(df['title_overlap'], df['query_length']) ]
    df['total_match_descr'] = [math.floor(x/y) for x,y in zip(df['descr_overlap'], df['query_length']) ]
    df['numbers?'] = [sum(s.isdigit() for s in x.split()) for x in data['search_term']]
    df['number_st_rel'] = [y/x for x,y in zip(df['query_length'],df['numbers?'])]
    df['query_length_1'] = [1 if x==1 else 0 for x in df['query_length']]
    df['query_length_chars'] = [len(x) for x in data['search_term']]
    # feature query term 80% composed of numbers -> not so good
    # edit distance 
    # character n-grams 
    # length of query = 1 -> not really helpful 
    # querry length in characters -> works very good
    # weight last word higher?,, parser?
    # similarity with relevant queries? 
    
    df = pd.DataFrame({
        'descr_overlap': df['descr_overlap'],
        'title_overlap': df['title_overlap'],
        'descr_match': df['descr_match'],
        'title_match': df['title_match'],
        'brand_match': df['brand_match'],
        'attribute_overlap': df['attribute_overlap'],
        'query_length': df['query_length'],
        'total_match_title': df['total_match_title'],
          #'numbers?' : df['numbers?'],
        'total_match_descr': df['total_match_descr'],   
        'number_st_rel': df['number_st_rel'],
          #'query_length_1': df['query_length_1'],
        'query_length_chars': df['query_length_chars'],    
        #'attribute_match':df['attribute_match'],
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


# loading the stemmed data 
df_train = pd.read_csv('Data\\stemmed\\train.csv', encoding="ISO-8859-1")
df_description = pd.read_csv('Data\\stemmed\\product_descriptions.csv', encoding="ISO-8859-1")
df_attributes = pd.read_csv('Data\\stemmed\\attributes.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('Data\\stemmed\\test.csv', encoding="ISO-8859-1")

# check the data after stemming 
df_attributes.iloc[0:50] # looks fine 

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

# converting the colors to 3-grams 
#len(df_train['joined_attributes'])
def n_gram(attribute, data, n_gram_attr, n):
    data[str(n_gram_attr)] = data[str(attribute)]
    
    
    for idx in range(len(data[str(attribute)])):
        current = data[str(attribute)].iloc[idx]
        result = []
        sentence = str(current).split()
        for word in sentence:
            chars = [c for c in word]
            # falls chars < n dann fallunterscheidung
            if len(chars) < n:
                grams = chars
            else:
                grams = ngrams(chars,n)

            for t in grams:
                result.append(''.join(t)) 
        data.loc[idx,(str(n_gram_attr))] = " ".join(result) 
    
	
	
# example usage

n_gram('search_term', df_train, 'search_term_ngram', 3 )

df_test = pd.merge(df_test, df_description, how='left', on='product_uid')
df_test = pd.merge(df_test, brands, how='left', on='product_uid')
df_test.drop('name',inplace=True,axis=1)
df_test.columns = df_test.columns.str.replace('value','brand')

df_test = pd.merge(df_test, groupeddf, how='left', on='product_uid')

id_test = df_test['id']

# extract the features on the training and test data set
# rule based classification
def rule_based(x_test):
    
    y_pred = np.zeros(len(x_test))
    final_score = np.zeros(len(x_test))

    #for idx in range(len(x_test)):

    score_attr_overlap = [1 if x!= 0 else 0 for x in x_test['attribute_overlap']]
    score_brand_match = [1 if x!= 0 else 0 for x in x_test['brand_match']]
    score_descr_overlap = [1 if x!= 0 else 0 for x in x_test['descr_overlap']]
    score_title_overlap = [1 if x!= 0 else 0 for x in x_test['title_overlap']]
    score_total_match = [1 if x!= 0 else 0 for x in x_test['total_match_title']]
    score_total_match2 = [1 if x!= 0 else 0 for x in x_test['total_match_descr']]
    for idx in range(len(x_test)):

        final_score[idx] = score_attr_overlap[idx] + score_brand_match[idx] + score_descr_overlap[idx] + score_title_overlap[idx]

    for idx in range(len(y_pred)):
        #if x_test['query_length'].iloc[idx] == 1:
         #   y_pred[idx] = 0
        #else:
        #    y_pred[idx] = 1.67
        #if final_score[idx] == 3:
        #    y_pred[idx] = 2.5
        if final_score[idx] == 0:
            y_pred[idx] = 0

            
    # extract the rest of the test data 

    indices = np.where(y_pred== 0)[0]
    ind = [x for x in indices]
    x_test_remain = x_test.iloc[ind]
    
    return y_pred, x_test_remain, ind
    #return y_pred
	
### train only on the xtremes!

def train_on_extremes(x_train,y_train):
    train_ind1 = np.where(y_train >= 1.5)[0]
    train_ind2 = np.where(y_train <= 2.5)[0]
    #train_ind = train_ind1
    train_ind = np.concatenate((train_ind1, train_ind2))
    new_x_train = x_train.iloc[train_ind]
    new_y_train = y_train[train_ind]
    
    return new_x_train, new_y_train
	
# discretizaton

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def discretizer(y_pred):
    result = np.ones(len(y_pred))
    compare = [1,1.33,1.67,2,2.33,2.67,3]
    for idx in range(len(y_pred)):
        current = round(3*(y_pred[idx]))/3;
        nearest_val = find_nearest(compare, current)
        result[idx] = nearest_val
        
    return result


# Cross-validation setup
N = df_train.shape[0]
kf = cross_validation.KFold(N, n_folds=10, random_state=2016)
avg_rmse = 0.
for train, test in kf:
    train_set = df_train.loc[train]
    test_set  = df_train.loc[test]
    x_train, x_test, y_train, y_test = extract_features(train_set, test_set)
    
    #frames = [x_train, x_train]
    #x_train = pd.concat(frames)
    #y_train = np.concatenate((y_train,y_train))
    
    #y_pred, x_test, indices = rule_based(x_test)
    
    #y_pred = rule_based(x_test)
    #x_train, y_train = train_on_extremes(x_train, y_train)
    
    # max_features = 7
    clf = RandomForestRegressor(n_estimators = 50, max_depth = 11, random_state = 2016, n_jobs = -1)
    #clf = MLPRegressor()
    #clf = ExtraTreesRegressor(n_estimators = 50, max_depth = 11,n_jobs = -1)
    #clf = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01,max_depth=4, min_samples_split = 1, random_state=2016, loss='ls')
    print("Fitting random forest regressor")
    clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    #y_pred[indices] = y_pred2 
    
    #y_pred = discretizer(y_pred)
    rmse_fold = rmse(y_test, y_pred)
    print(rmse_fold)
    avg_rmse += rmse_fold
avg_rmse = avg_rmse/10
print("Avg rmse: " + str(avg_rmse))



#x_train, x_test, y_train = extract_features(df_train, df_test)
#clf = RandomForestRegressor(n_estimators=30, random_state=26, verbose=True)
#clf.fit(x_train, y_train)
#y_pred2 = clf.predict(x_test_remain)

#pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv', index=False)

sum(x_train['query_length']==0)

ind = np.where(abs(y_pred - y_test) >= 1)[0]
print(len(ind))
print(len(y_pred))
print(len(test_set))

relevant_results = test_set.iloc[ind]
#print(y_pred[ind][120:122])
relevant_results[100:200]


print(clf.feature_importances_)
print(clf.n_features_)

print(y_pred[1:100])
print(y_test[1:100])
