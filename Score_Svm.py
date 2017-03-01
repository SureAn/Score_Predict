print(__doc__)

import matplotlib.pyplot as plt
from Load_Data import *
from Data_Preprocess import *
from Train_algo import *
from Predict_Data import *
from Test_Analysis import *

#load train data
X = load_data("train_data_X.csv")
y = load_data("train_data_Y.csv")
#train data preprocess
#X = data_preprocess(X)
#y = data_preprocess(y)
#fit support vector machine,return model
model = train_svm(X,y,30,0.01)
#load text data
y1 = load_data("test_data_Y.csv")
X1 = load_data("test_data_X.csv")
#test data preprocess
#X1 = data_preprocess(X)
#y1 = data_preprocess(y)
#test data predict
predicted = predict_data(model,X1)
#analysis
test_analysis(predicted,y1)
#lw = 2
#plt.scatter(X1, y1, color='darkorange', label='data')
#plt.hold('on')
#plt.plot(X1, predicted, color='navy',  label='RBF model')
#plt.xlabel('data')
#plt.ylabel('target')
#plt.title('Support Vector Regression')
#plt.legend()
#plt.show()
