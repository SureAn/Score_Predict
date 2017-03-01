from sklearn import preprocessing

#data preprocess :standardize and normalize the data
def data_preprocess(X):
    X = preprocessing.normalize(preprocessing.scale(X))
    return X