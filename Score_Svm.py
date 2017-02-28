print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing,metrics
from sklearn.svm import SVC
import csv
#from sklearn.linear_model import SGDClassifier


#trainXreader = csv.reader(open("train_data_X.csv"))
trainXreader = np.loadtxt(open("train_data_X.csv","rb"),delimiter=",",skiprows=0)
X = trainXreader[:,1:8]

trainYreader = np.loadtxt(open("train_data_Y.csv","rb"),delimiter=",",skiprows=0)
y = trainYreader[:,1]

# normalize the data attributes
#normalized_X = preprocessing.normalize(X)
# standardize the data attributes
#standardized_X = preprocessing.scale(X)

model = SVC(C=30,class_weight={0:7},gamma=0.03)
model.fit(X, y)
print(model)
trainY1reader = np.loadtxt(open("test_data_Y.csv","rb"),delimiter=",",skiprows=0)
y1 = trainY1reader[:,1]
expected = y1
trainX1reader = np.loadtxt(open("test_data_X.csv","rb"),delimiter=",",skiprows=0)
X1 = trainX1reader[:,1:8]
predicted = model.predict(X1)
print(predicted)
np.savetxt('predict.csv', predicted, delimiter = ',')
#计算整体准确率
accuracy = metrics.accuracy_score(y1, predicted)
print('accuracy: %.2f%%' % (100 * accuracy))

#计算“0”样本的准确率
count_y=0
count_P=0
for j in range(1,298):
    if y1[j]==0:
        count_y+=1
        if predicted[j]==0:
            count_P+=1
accuracy_really = count_P/count_y
print('accuracy_really: %.2f%%' % (100 * accuracy_really))
#lw = 2
#plt.scatter(X1, y1, color='darkorange', label='data')
#plt.hold('on')
#plt.plot(X1, predicted, color='navy',  label='RBF model')
#plt.xlabel('data')
#plt.ylabel('target')
#plt.title('Support Vector Regression')
#plt.legend()
#plt.show()
