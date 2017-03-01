import numpy as np
def predict_data(model,X):
    predicted = model.predict(X)
    print(predicted)
    np.savetxt('predict.csv', predicted, delimiter=',')
    return predicted
