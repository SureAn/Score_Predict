from sklearn.svm import SVC

def train_svm(X,y,C,gamma):
    model = SVC(C=C,class_weight={0:8},gamma=gamma)
    model.fit(X,y)
    print(model)
    return model
