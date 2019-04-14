#!/usr/bin/env python
import math
import operator
from sklearn.metrics import mean_squared_error
import pandas

class kNN(object):

    def __init__(self, x, y, k, weighted=False):
        assert (k <= len(x)
                ), "k cannot be greater than training_set length"
        self.__x = x
        self.__y = y
        self.__k = k
        self.__weighted = weighted

    @staticmethod
    def __euclidean_distance(x1, y1, x2, y2): #use L2 distance as the metric function
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    @staticmethod
    def gaussian(dist, sigma=1): # for weighted k-NN use gaussion distribution as the weight
        return 1./(math.sqrt(2.*math.pi)*sigma)*math.exp(-dist**2/(2*sigma**2))

    def predict(self, test_set): # use the average of the k closest sample as the prediction 
        predictions = []
        for i, j in test_set.values:
            distances = []
            for idx, (l, m) in enumerate(self.__x.values):
                dist = self.__euclidean_distance(i, j, l, m)
                distances.append((self.__y[idx], dist))
            distances.sort(key=operator.itemgetter(1)) #sort the get the k closest smaples
            v = 0
            total_weight = 0
            for i in range(self.__k):
                weight = self.gaussian(distances[i][1])
                if self.__weighted: # wighted k-NN
                    v += distances[i][0]*weight
                else:
                    v += distances[i][0] # sample k-NN
                total_weight += weight
            if self.__weighted:
                predictions.append(v/total_weight)
            else:
                predictions.append(v/self.__k)
        return predictions


training_data = pandas.read_csv("auto_train.csv")
x = training_data.iloc[:,:-1]
y = training_data.iloc[:,-1]

test_data = pandas.read_csv("auto_test.csv")
x_test = test_data.iloc[:,:-1]
y_test = test_data.iloc[:,-1]

for k in [1, 3, 20]:
    classifier = kNN(x,y, k) #simple k-NN
    #classifier = kNN(x,y, k,weighted=Ture)  #Using weighted k-NN we obtained better performance than with simple k-NN.
    pred_test = classifier.predict(x_test)

    test_error = mean_squared_error(y_test, pred_test) #use mean_squared_error to evaluate the performance
    print("Test error with k={}: {}".format(k, test_error * len(y_test)/2))