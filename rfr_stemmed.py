import pandas as pd
from sklearn import cross_validation
import os.path
from feature_extraction.extraction import FeatureExtractor
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVR, SVR
from scipy.optimize import minimize

def rmse(true, test):
    return mean_squared_error(true, test)**0.5

def get_target_values(train, test):
    y_train = train['relevance'].values
    if 'relevance' in test:
        y_test = test['relevance'].values
        return y_train, y_test
    return y_train

def run_cross_val(df_train, K, w):
    N = df_train.shape[0]

    # Cross-validation setup
    kf = cross_validation.KFold(N, n_folds=K, shuffle=True)
    avg_rmse = 0.

    for train, test in kf:
        train_set = df_train.loc[train]
        test_set = df_train.loc[test]
        x_train = df_x_train.iloc[train]
        x_test = df_x_train.iloc[test]

        y_train, y_test = get_target_values(train_set, test_set)

        clfs = train_classifiers(x_train, y_train)
        y_pred = predict_test(clfs, x_test)
        y_pred = np.dot(y_pred, w)

        rmse_fold = rmse(y_test, y_pred)
        print(rmse_fold)
        avg_rmse += rmse_fold

    avg_rmse = avg_rmse/K
    print("Avg rmse: " + str(avg_rmse))
    
def train_classifiers(x_train, y_train):    
    clfs = []
    
    #Random forest
    clf_rfr = RandomForestRegressor(n_estimators=100, max_depth=11, n_jobs=-1)
    clf_rfr.fit(x_train, y_train)
    clfs.append(clf_rfr)
    
    #Random forest stumps
    clf_rfr2 = RandomForestRegressor(n_estimators=50, max_depth=1, n_jobs=-1)
    clf_rfr2.fit(x_train, y_train)
    clfs.append(clf_rfr2)    
    
    #AdaBoost
    clf_ada = AdaBoostRegressor(n_estimators=200, learning_rate=1)
    clf_ada.fit(x_train, y_train)
    clfs.append(clf_ada)
    
    #Logistic Regression
    clf_log = Ridge()
    clf_log.fit(x_train, y_train)
    clfs.append(clf_log)
    
    #Linear SVM
    clf_linSVM = LinearSVR()
    clf_linSVM.fit(x_train, y_train)
    clfs.append(clf_linSVM)
    
    
    return clfs
    
def predict_test(clfs, test):
    y_pred = np.zeros((len(test),1))
    
    for clf in clfs:
        pred = clf.predict(test)
        y_pred = np.c_[y_pred,pred]
        
    y_pred = y_pred[:, 1:]
    
    #Determining weights
    #w = np.ones(len(clfs))/len(clfs)    #Avg of all ensembles
    #w = np.array([0.8, 0.1, 0.0, 0.1])
    
    return y_pred

def find_ensemble_weights(clfs, y_pred, y_test):
    #w = np.ones(len(clfs))/len(clfs)
    w = np.array([1./len(clfs)] * len(clfs))
    
    cons = ({'type':'eq', 'fun': lambda w: 1-np.sum(w)})
    bounds = [(0,1)]*len(w)
    
    res = minimize(rmse_w, w, args=(y_test, y_pred), bounds=bounds, constraints=cons)
    
    print("Ensemble weights: ", res['x'])
    return res['x']

def rmse_w(w, y_test, y_pred):
    test = np.dot(y_pred, w)
    
    return mean_squared_error(y_test, test)**0.5


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

	
fext = FeatureExtractor(df_description, df_attributes, verbose=True)

df_train = fext.extractTextualFeatures(df_train)
df_x_train = fext.extractNumericalFeatures(df_train)

#Finding weights for ensemble using 20% of training data
N = len(df_train)
i_test = list(np.random.choice(range(N), size=int(N*0.2), replace=False))
i_train = list(set(range(N))-set(i_test))

train_set = df_train.loc[i_train]
test_set  = df_train.loc[i_test]

x_train_w = df_x_train.iloc[i_train]
x_test_w = df_x_train.iloc[i_test]
y_train_w, y_test_w = get_target_values(train_set, test_set)

clfs = train_classifiers(x_train_w, y_train_w)
y_pred = predict_test(clfs, x_test_w)
w = find_ensemble_weights(clfs, y_pred, y_test_w)


#Run Kfold CV
run_cross_val(df_train, 5, w)


df_test = fext.extractTextualFeatures(df_test)
x_test  = fext.extractNumericalFeatures(df_test)
x_train = df_x_train

id_test = df_test['id']


y_train = get_target_values(df_train, df_test)
clfs = train_classifiers(x_train, y_train)
y_pred = predict_test(clfs, x_test)
y_pred = np.dot(y_pred, w)


pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv', index=False)