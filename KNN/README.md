# KNN for classification task

## data description

The problem is comprised of 150 observations of iris flowers from three different species. There are 4 measurements of given flowers: sepal length, sepal width, petal length and petal width, all in the same unit of centimeters. The predicted attribute is the species, which is one of setosa, versicolor or virginica.

It is a standard dataset where the species is known for all instances. As such we can split the data into training and test datasets and use the results to evaluate our algorithm implementation.

**data file**: iris.data (*provided already*)

## code explain

### Handle data

The first thing we need to do is load our data file. The data is in CSV format without a header line or any quotes. We can open the file with the open function and read the data lines using the reader function in the [csv](https://docs.python.org/2/library/csv.html) module.

```&#39;
import csv
with open('iris.data','rb') as csvfile:
	lines=csv.reader(csvfile)
	for row in lines:
		print ','.join(row)	
```

Next we need to split the data into a training dataset that kNN can use to make predictions and a test dataset that we can use to evaluate the accuracy of the model.

We first need to convert the flower measures that were loaded as strings into numbers that we can work with. Next we need to split the data set randomly into train and datasets. A ratio of 67/33 for train/test is a standard ratio used.

Pulling it all together, we can define a function called **loadDataset** that loads a CSV with the provided filename and splits it randomly into train and test datasets using the provided split ratio.

### Similarity

In order to make predictions we need to calculate the similarity between any two given data instances. This is needed so that we can locate the k most similar data instances in the training dataset for a given member of the test dataset and in turn make a prediction.

Given that all four flower measurements are numeric and have the same units, we can directly use the Euclidean distance measure. This is defined as the square root of the sum of the squared differences between the two arrays of numbers (read that again a few times and let it sink in).

Additionally, we want to control which fields to include in the distance calculation. Specifically, we only want to include the first 4 attributes. One approach is to limit the euclidean distance to a fixed length, ignoring the final dimension.

Putting all of this together we can define the **euclideanDistance** function .

### Neighbors

Now that we have a similarity measure, we can use it collect the k most similar instances for a given unseen instance.

This is a straight forward process of calculating the distance for all instances and selecting a subset with the smallest distance values.

The **getNeighbors** function returns k most similar neighbors from the training set for a given test instance (using the already defined **euclideanDistance** function)

### Response

Once we have located the most similar neighbors for a test instance, the next task is to devise a predicted response based on those neighbors.

We can do this by allowing each neighbor to vote for their class attribute, and take the majority vote as the prediction.

**getResponse** function for getting the majority voted response from a number of neighbors. It assumes the class is the last attribute for each neighbor.

### Accuracy

We have all of the pieces of the kNN algorithm in place. An important remaining concern is how to evaluate the accuracy of predictions.

An easy way to evaluate the accuracy of the model is to calculate a ratio of the total correct predictions out of all predictions made, called the classification accuracy.

The **getAccuracy** function that sums the total correct predictions and returns the accuracy as a percentage of correct classifications.

# KNN for regression task

## data description

To test our k-NN implementation we will perform experiments using a version of the automobile dataset from the UC Irvine Repository. The problem will be to predict the miles per gallon (mpg) of a car, given its displacement and horsepower. Each example in the dataset corresponds to a single car.

```
Number of Instances: 291 in the training set, 100 in the test set
Number of Attributes: 2 continous input attributes, one continuous output
```

Attribute Information:

```
1. displacement:  continuous 
2. horsepower:    continuous
3. mpg:           continuous (output)
```

The following is an extract of how the dataset looks like:

```
displacement,horsepower,mpg
307,130,18
350,165,15
318,150,18
304,150,16
302,140,17
429,198,15
454,220,14
440,215,14
455,225,14
```

### Read the data

First, we read the data using pandas.

```
import pandas

training_data = pandas.read_csv("auto_train.csv")
x = training_data.iloc[:,:-1]
y = training_data.iloc[:,-1]

test_data = pandas.read_csv("auto_test.csv")
x_test = test_data.iloc[:,:-1]
y_test = test_data.iloc[:,-1]
```

## code explain

### distance

For our purposes we will adopt Euclidean distance and since our dataset is made of two attributes we can use the following function.

```
    @staticmethod
    def __euclidean_distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
```

### Weighted distance

Instead of computing an average of the **k** neighbors, we can compute a weighted average of the neighbors. A common way to do this is to weight each of the neighbors by a factor of **1/d**, where **d** is its distance from the test example.  

For our implementation, we chose to use weighted distance according to a paper[1](https://www.antoniomallia.it/on-implementing-k-nearest-neighbor-for-regression-in-python.html#fn:fnSarma) which proposes another improvement to the basic k-NN where the weights to nearest neighbors are given based on **Gaussian distribution**.

```
    @staticmethod
    def gaussian(dist, sigma=1):
        return 1./(math.sqrt(2.*math.pi)*sigma)*math.exp(-dist**2/(2*sigma**2))
```

### Initialization

Given a training set, we need first to store it as we will use it at prediction time. Clearly, **k cannot be bigger than the training set itself.**

```
class kNN(object):

    def __init__(self, x, y, k, weighted=False):
        assert (k <= len(x)
                ), "k cannot be greater than training_set length"
        self.__x = x
        self.__y = y
        self.__k = k
        self.__weighted = weighted
```

### Prediction

Predicting the output for a new example **x** is conceptually trivial. All we need to do is:

- iterate through the examples
- measure the distance from each one to **x**
- keep the best **k**

Depending if we are doing classification or regression we would treat those **k** examples differently. In this case, we will do regression, so our prediction will be just the average of the samples.

```
    def predict(self, test_set):
        predictions = []
        for i, j in test_set.values:
            distances = []
            for idx, (l, m) in enumerate(self.__x.values):
                dist = self.__euclidean_distance(i, j, l, m)
                distances.append((self.__y[idx], dist))
            distances.sort(key=operator.itemgetter(1))
            v = 0
            total_weight = 0
            for i in range(self.__k):
                weight = self.gaussian(distances[i][1])
                if self.__weighted:
                    v += distances[i][0]*weight
                else:
                    v += distances[i][0]
                total_weight += weight
            if self.__weighted:
                predictions.append(v/total_weight)
            else:
                predictions.append(v/self.__k)
        return predictions
```

If we are happy with an implementation that takes **O(N)** execution time, then that is the end of the story. If not, there are possible optimization using indexes based on additional data structures, i.e. k-d trees or hash tables, which I might write about in the future.

### simple k-NN result 

Using the data in the training set, we predicted the output for each example in the test, for k=1, k=3, and k=20. Reported the **squared error** on the test set. As we can see the test error goes down while increasing k.

```
from kNN import kNN
from sklearn.metrics import mean_squared_error

for k in [1, 3, 20]:
    classifier = kNN(x,y, k)
    pred_test = classifier.predict(x_test)

    test_error = mean_squared_error(y_test, pred_test)
    print("Test error with k={}: {}".format(k, test_error * len(y_test)/2))
```

> Output:
>
> Test error with k=1: 2868.0049999999997
> Test error with k=3: 2794.729999999999
> Test error with k=20: 2746.1914125

### Weighted k-NN result 

Using weighted k-NN we obtained better performance than with simple k-NN.

```
from kNN import kNN

for k in [1, 3, 20]:
    classifier = kNN(x,y, k, weighted=True)
    pred_test = classifier.predict(x_test)

    test_error = mean_squared_error(y_test, pred_test)
    print("Test error with k={}: {}".format(k, test_error * len(y_test)/2))
```

> Output:
>
> Test error with k=1: 2868.005
> Test error with k=3: 2757.3065023859417
> Test error with k=20: 2737.9437262401907