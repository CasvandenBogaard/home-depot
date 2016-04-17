import pandas as pd
from sklearn import cross_validation
import os.path
from feature_extraction.extraction import FeatureExtractor
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def rmse(true, test):
    return mean_squared_error(true, test)**0.5

def get_target_values(train, test):
    y_train = train['relevance'].values
    if 'relevance' in test:
        y_test = test['relevance'].values
        return y_train, y_test
    return y_train

def run_cross_val(df_train, K, w=0, find_weights=False):
    N = df_train.shape[0]

    # Cross-validation setup
    kf = cross_validation.KFold(N, n_folds=K, shuffle=True)
    avg_rmse = 0.
    
    w_avg = []

    for train, test in kf:
        train_set = df_train.loc[train]
        test_set = df_train.loc[test]
        x_train = df_x_train.iloc[train]
        x_test = df_x_train.iloc[test]

        y_train, y_test = get_target_values(train_set, test_set)

        clfs, clf_feats = train_classifiers(x_train, y_train)
        y_pred = predict_test(clfs, clf_feats, x_test)
        
        if find_weights:
            w = find_ensemble_weights(clfs, y_pred, y_test)
            
            if len(w_avg)==0:
                w_avg = np.array(w)
            else:
                w_avg += np.array(w)
            
        y_pred = np.dot(y_pred, w)

        rmse_fold = rmse(y_test, y_pred)
        print("RMSE: ", rmse_fold)
        avg_rmse += rmse_fold

    avg_rmse = avg_rmse/K
    print("Avg rmse: " + str(avg_rmse))
    
    return np.array(w_avg)/len(kf)
    
def train_classifiers(x_train, y_train):
    print("Xtrain shape: {}".format(x_train.shape))
    clfs = []
    clf_feats = []
    labels = list(x_train)
    
    #Random forest
    #Kaggle score: 0.47834
    clf_rfr = RandomForestRegressor(n_estimators=5000, max_depth=11, n_jobs=-1)
    features = []
    x_feats = keep_features(x_train, features)
    clf_rfr.fit(x_feats, y_train)
    

    clfs.append(clf_rfr)
    clf_feats.append(features)
    
    
    #Random forest
    #Kaggle score: 0.47834
    #clf_rfr2 = RandomForestRegressor(n_estimators=100, max_depth=11, n_jobs=-1)
    #features = ['color_match', 'brand_match', 'query_length']
    #x_feats = keep_features(x_train, features)
    #clf_rfr2.fit(x_feats, y_train)
    
    #clfs.append(clf_rfr2)
    #clf_feats.append(features)


    #AdaBoost
    #clf_ada = AdaBoostRegressor(n_estimators=200, learning_rate=1)
    #features = []
    #x_feats = keep_features(x_train, features)
    #clf_ada.fit(x_feats, y_train)
    
    #clfs.append(clf_ada)
    #clf_feats.append(features)
    
    #Logistic Regression
    #clf_log = Ridge()
    #features = []
    #x_feats = keep_features(x_train, features)
    #clf_log.fit(x_feats, y_train)
    
    #clfs.append(clf_log)
    #clf_feats.append(features)
    
    #Linear SVM
    #clf_svm = LinearSVR()
    #features = []
    #x_feats = keep_features(x_train, features)
    #clf_svm.fit(x_feats, y_train)
    
    #clfs.append(clf_svm)
    #clf_feats.append(features)
    
    #1NN Regressor 
    #Kaggle score: 0.65825
    #clf_knn1 = KNeighborsRegressor(n_neighbors=1)
    #features = []
    #x_feats = keep_features(x_train, features)
    #clf_knn1.fit(x_feats, y_train)
    
    #clfs.append(clf_knn1)
    #clf_feats.append(features)
    
    #5NN Regressor
    #Kaggle score: 0.52425
    #clf_knn2 = KNeighborsRegressor(n_neighbors=5)
    #features = []
    #x_feats = keep_features(x_train, features)
    #clf_knn2.fit(x_feats, y_train)
    
    #clfs.append(clf_knn2)
    #clf_feats.append(features)
    
    #Bayesian Ridge Regression
    #clf_br = BayesianRidge()
    #features = []
    #x_feats = keep_features(x_train, features)
    #clf_br.fit(x_feats, y_train)
    
    #clfs.append(clf_br)
    #clf_feats.append(features)
    
    #Gradient Boosting
    #Kaggle score: 0.47744
    clf_gb = GradientBoostingRegressor(n_estimators=7000)
    features = []
    x_feats = keep_features(x_train, features)
    clf_gb.fit(x_feats, y_train)
    
    clfs.append(clf_gb)
    clf_feats.append(features)
    
    #Bagging Regressor
    #clf_bag = BaggingRegressor()
    #features = []
    #x_feats = keep_features(x_train, features)
    #clf_bag.fit(x_feats, y_train)

    #clfs.append(clf_bag)
    #clf_feats.append(features)
    
    #Kernel Ridge Regression
    #clf_kr = KernelRidge()
    #features = []
    #x_feats = keep_features(x_train, features)
    #clf_kra.fit(x_feats, y_train)
    
    #clfs.append(clf_kr)
    #clf_feats.append(features)

    #Network regressor
    # clf_net = net
    # features = []
    # clfs.append(clf_net)
    # clf_feats.append(features)
    
    return clfs, clf_feats
    
def predict_test(clfs, clf_feats, test):
    y_pred = np.zeros((len(test),1))
    
    for clf, feats in zip(clfs, clf_feats):
        test_feats = keep_features(test, feats)
        print("testfeats")
        print(test_feats.shape)
        pred = clf.predict(test_feats)
        print(y_pred.shape)
        print(pred.shape)
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

def plotHistograms(y_pred):
    for i in range(len(y_pred[0])):
        plt.hist(y_pred[:, i])
        plt.xlim(1,3)
        plt.show()
    
    print(y_pred)
    
def keep_features(x_train, features):
    #Empty feature list: fit all features
    if features == []:
        return x_train
    else:
        x_feats = x_train
        feats = list(x_train)
        for f in features:
            feats.remove(f)
        
        for feat in feats:
            x_feats = x_feats.drop(feat, axis=1, inplace=False)  
    
        return x_feats


# use preprocessing.py if you want to stem
if (os.path.isdir('data/stemmed')):
    df_train = pd.read_csv('data/stemmed/train.csv', encoding="ISO-8859-1")
    df_description = pd.read_csv('data/stemmed/product_descriptions.csv', encoding="ISO-8859-1")
    df_attributes = pd.read_csv('data/stemmed/attributes.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('data/stemmed/test.csv', encoding="ISO-8859-1")
    
    df_train_unstemmed = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
    df_test_unstemmed = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
else:
    df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
    df_description = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
    df_attributes = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")

fext = FeatureExtractor(df_description, df_attributes, verbose=True, name="train")

df_train = fext.extractTextualFeatures(df_train)
df_x_train = fext.extractNumericalFeatures(df_train, df_train_unstemmed)
y_train = get_target_values(df_train, df_test)


# net = Network(df_x_train, y_train)


w = run_cross_val(df_train, 5, find_weights=True)

print("Avg of weights: ",w)

run_cross_val(df_train, 5, w=w)

fext = FeatureExtractor(df_description, df_attributes, verbose=True, name="test")
df_test = fext.extractTextualFeatures(df_test)
x_test  = fext.extractNumericalFeatures(df_test, df_test_unstemmed)

x_train = df_x_train

id_test = df_test['id']



y_train = get_target_values(df_train, df_test)
clfs, clf_feats = train_classifiers(x_train, y_train)
y_pred = predict_test(clfs, clf_feats, x_test)
y_pred = np.dot(y_pred, w)


pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv', index=False)