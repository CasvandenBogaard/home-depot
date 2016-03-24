import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import os.path
from feature_extraction.extraction import FeatureExtractor

def rmse(true, test):
    return mean_squared_error(true, test)**0.5

def get_target_values(train, test):
    y_train = train['relevance'].values
    if 'relevance' in test:
        y_test = test['relevance'].values
        return y_train, y_test
    return y_train

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

N = df_train.shape[0]

# Cross-validation setup
kf = cross_validation.KFold(N, n_folds=10, shuffle=True)
avg_rmse = 0

for train, test in kf:
    train_set = df_train.loc[train]
    test_set = df_train.loc[test]
    x_train = df_x_train.iloc[train]
    x_test = df_x_train.iloc[test]

    y_train, y_test = get_target_values(train_set, test_set)

    clf = RandomForestRegressor(n_estimators=100, max_depth=11, n_jobs=-1)
    print("Fitting random forest regressor")
    
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    names = list(x_train.columns.values)

    #uniques = np.unique(y_test)

    # results = []
    # for u in uniques:
    #     indexes = np.where(y_test==u)
    #     results.append(y_pred[indexes])
    #
    # plt.boxplot(results)
    # plt.plot(range(1, len(uniques) + 1), uniques, 'o')
    # plt.xticks(range(1, len(uniques) + 1), uniques)
    # plt.ylim((0,4))
    # plt.show()
    #
    # print(x_test.iloc[0])
    # print(clf.feature_importances_)

    rmse_fold = rmse(y_test, y_pred)
    print(rmse_fold)
    avg_rmse += rmse_fold

avg_rmse = avg_rmse/10
print("Avg rmse: " + str(avg_rmse))

df_test = fext.extractTextualFeatures(df_test)
x_test  = fext.extractNumericalFeatures(df_test)
x_train = df_x_train

id_test = df_test['id']


y_train = get_target_values(df_train, df_test)
clf = RandomForestRegressor(n_estimators=100, max_depth=11, n_jobs=-1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('results/submission.csv', index=False)