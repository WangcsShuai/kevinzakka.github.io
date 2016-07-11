---
layout: post
title: "A Complete Guide to K-Nearest-Neighbors with Applications in Python and R"
date: 2016-07-11
excerpt: "I'll introduce the intuition and math behind KNN, cover a real-life example, and explore the inner-workings of the algorithm by implementing the code from scratch."
tags: [KNN, machine learning, classification, neighbours]
comments: true
mathjax: true
---
This is an in-depth tutorial designed to introduce you to a simple, yet powerful classification algorithm called K-Nearest-Neighbors (KNN). We will go over the intuition and mathematical detail of the algorithm, apply it to a real-world dataset to see exactly how it works, and finally gain an intrinsic understanding of the algorithm by writing it from scratch in code. Finally, we will look at ways in which we can improve KNN.

## Table of Contents

1. [Introduction](#introduction)
2. [What is KNN?](#what-is-knn)
3. [How does KNN work?](#how-does-knn-work)
4. [More on K](#more-on-k)
5. [Exploring KNN in Code](#exploring-knn-in-code)
6. [Parameter Tuning with Cross Validation](#parameter-tuning-with-cross-validation)
7. [Writing our Own KNN from Scratch](#writing-our-own-knn-from-scratch)
8. [Pros and Cons of KNN](#pros-and-cons-of-knn)
9. [Improvements](#improvements)
10. [Tutorial Summary](#tutorial-summary)

## Introduction

The KNN algorithm is a robust and versatile classifier that is often used to provide a benchmark for more complex classifiers such as Artificial Neural Networks (ANN) and Support Vector Machines (SVM). Despite its simplicity, KNN can outperform more powerful classifiers and is used in a variety of applications such as economic forecasting, data compression and genetics. For example, KNN was leveraged in a study of functional genomics for the assignment of genes based on their expression profiles.

## What is KNN?
Let's first start by establishing some definitions and notations. We will use $$x$$ to denote a *feature* (also known as: predictor, attribute) and $$y$$ to denote the *target* (also known as: response, label, class)  we are trying to predict.

KNN falls in the __supervised learning__ family of algorithms. Informally, this means that we are given a labelled dataset consiting of training observations $$(x,y)$$ and want to capture the relationship between $$x$$ and $$y$$. More formally, our goal is to learn a function $$h : X  → Y$$ so that given any unseen observation $$x$$, $$h(x)$$ can confidently predict the corresponding output $$y$$.

The KNN classifier is also a __non parametric__ and __instance-based__ learning algorithm. Non-parametric means it makes no explicit assumptions about the functional form of h, avoiding the danger of mismodeling the underlying distribution of the data. Furthermore, instance-based means it defers processing of the training data to the testing phase. So only when a query is made (i.e. when we ask it to predict a label given an input) will KNN run through the whole data and spit out the response. You can probably already tell this has a huge disadvantage both time and memory wise!

## How does KNN work?
Given a positive integer K (usually odd to avoid tie situations), an unseen observation $$x$$ and a similarity metric $$d$$, KNN essentially boils down to forming a majority vote between the K most similar instances to $$x$$. Similarity is equivalent to taking the distance between two given data points. A popular choice for $$d$$ is the Euclidean distance

$$d(x, x') = \sqrt{\left(x_1 - x'_1 \right)^2 + \left(x_2 - x'_2 \right)^2 + \dotsc + \left(x_n - x'_n \right)^2}$$

but other measures can be more suitable for a given setting and include the Manhattan, Chebyshev and Hamming distance.

More formally, in the classification setting, KNN classifier works in two steps:

- It runs through the whole dataset computing $$d$$ between $$x$$ and each training observation. We'll call the K points in the training data that are closest to $$x$$ the set $$\mathcal{A}$$.

- It then estimates the conditional probability for each class, that is, the fraction of points in $$\mathcal{A}$$ with that given class label. (Note $$I(x)$$ is the indicator function which evaluates to $$1$$ when the argument $$x$$ is true and $$0$$ otherwise)

$$P(y = j | X = x) = \frac{1}{K} \sum_{i \in \mathcal{A}} I(y^{(i)} = j)$$

Finally, our input $$x$$ gets assigned to the class with the largest probability.

## More on K
Let's talk a bit about the variable K. Like most machine learning algorithms, the K in KNN is a hyperparameter that we must tune in order to get the best possible classifier for a given dataset. Essentially, the choice of K has an effect on the shape of the prediction curves (i.e. decision boundary) of our classifier. 

In fact, when K is small, we are restraining the region of a given prediction and forcing our classifier to be “more blind” to the overall distribution. A small value for K provides the most flexible fit, which will have low bias but high variance. Graphically, our decision boundary will be more jagged.

<img src="/assets/1nearestneigh.png">

On the other hand, a higher K averages more voters in each prediction and hence is more resilient to outliers. Larger values of K will have smoother decision boundaries which means lower variance but increased bias.

<img src="/assets/20nearestneigh.png">

(If you want to learn more about the bias-variance tradeoff, check out [Scott Roe's Blog post](http://scott.fortmann-roe.com/docs/BiasVariance.html). You can mess around with the value of K and watch the decision boundary change!)

## Exploring KNN in Code
<img src="/assets/flower.jpg">

Without further ado, let's see how KNN can be leveraged in Python for a classification problem. We’re gonna head over to the UC Irvine Machine Learning Repository, an amazing source for a variety of free and interesting data sets.  

The data set we'll be using is the [Iris Flower Dataset](https://archive.ics.uci.edu/ml/datasets/Iris) (IFD) which was first introduced in 1936 by the famous statistician Ronald Fisher and consists of 50 observations from each of three species of Iris (*Iris setosa, Iris virginica and Iris versicolor*). Four features were measured from each sample: the length and the width of the sepals and petals. Our goal is to train the KNN algorithm to be able to distinguish the species from one another given the measurements of the 4 features.

Go ahead and `Download Data Folder > iris.data` and save it in the directory of your choice.

The first thing we need to do is load the data set. It is in CSV format without a header line so we'll use pandas' `read_csv` function. 

```python
# loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# define column names
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# loading training data
df = pd.read_csv('/Users/kevin/Desktop/Blog/iris.data.txt', header=None, names=names)
df.head()
```
Next, it would be cool if we could examine the data before rushing into classification so that we can have a deeper understanding of the problem at hand. R has some beautiful visualization tools, so we'll be using it to create 2 quick scatter plots of __sepal width vs sepal length__ and __petal width vs petal length__. 

```r
# loading packages
library(ggvis)

# sepal width vs sepal length
iris %>% 
  ggvis(~Sepal.Length, ~Sepal.Width, fill = ~Species) %>%
  layer_points()
  
# petal width vs petal length
iris %>%
  ggvis(~Petal.Length, ~Petal.Width, fill = ~Species) %>%
  layer_points()
```
Note that we've accessed the `iris` dataframe which comes preloaded in R by default.

<img src="/assets/sep_plot.png">

<img src="/assets/pet_plot.png">

A quick study of the above graphs reveals some strong classification criterion. We observe that setosas have small petals, versicolor have medium sized ones and virginica have relatively larger petals. Furthermore, setosas seem to have shorter and wider sepals than the other two classes. Hence, without even using an algorithm, we can intuitevely construct a classifier that can do very well on the dataset.

The next step in our pipeline is to create our design matrix $$X$$ and our target vector $$y$$ as well as split our data into training and test sets. We will do so with the following code:

```python
# loading library
from sklearn.cross_validation import train_test_split

# create design matrix X and target vector y
X = np.array(df.ix[:, 0:4]) 	# end index is exclusive
y = np.array(df['class']) 	# another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

Now we define our classifer, in this case KNN, fit it to our training data and evaluate its accuracy. In this case, we'll be using an arbitrary K but we will see later on how cross validation can be used to find its optimal value.

```python
# loading library
from sklearn.neighbors import KNeighborsClassifier

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print accuracy_score(y_test, pred)
``` 

## Parameter Tuning with Cross Validation
At this point, you're probably wondering how to pick K. In order to "tune" the hyperparameter K so that we can obtain the best possible classifier, we're gonna use a resampling method called k-fold cross validation. (Note that the k in k-fold it totally unrelated to the K in KNN!)

<img src="/assets/k_fold_cv.jpg">

As seen above, k-fold cross validation involves randomly dividing the training set into k groups, or folds, of approximately equal size. The first fold is treated as a validation set, and the method is fit on the remaining k − 1 folds. The error metric (i.e. misclassification rate for classification and mean squared error for regression) is then computed on the observations in the held-out fold. This procedure is repeated k times; each time, a different group of observations is treated as a validation set. This process results in k estimates of the test error which are then averaged out.

Let's go ahead and perform 10-fold cross validation on our dataset using a generated list of odd K's ranging from 1 to 50.

```python
# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
```

Now, we determine the best K and graph the misclassification error versus the different values of K.

```python
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print "The optimal number of neighbors is %d" % optimal_k

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
```

<img src="/assets/cv_knn.png">

Our best K turns out to be 7.

## Writing our Own KNN from Scratch
So far, we've studied how KNN works and seen how we can use it for a classification task using scikit-learn's generic pipeline (i.e. input, instantiate train, predict and evaluate). Now, it's time to delve deeper into KNN by trying to code it ourselves from scratch.

A machine learning algorithm usually consists of 2 main blocks: 

- a __training__ block that takes as input the training data $$X$$ and the corresponding target $$y$$ and outputs a learned model $$h$$. 

- a __predict__ block that takes as input new and unseen observations and uses the function $$h$$ to output their corresponding responses.
 
In the case of KNN, which as discussed earlier, is a lazy algorithm, the training block reduces to just memorizing the training data. Let's go ahead a write a python method that does so.

```python
def train(X_train, y_train):
	# do nothing 
	return
```

Phew, that was hard! Now we need to write the predict method which must do the following: it needs to compute the euclidean distance between the "new" observation and all the data points in the training set. It must then select the K nearest ones and perform a majority vote. It then assigns the corresponding label to the observation. Let's go ahead and write that.

```python
def predict(X_train, y_train, x_test, k):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# first we compute the euclidean distance
		distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
		# add it to list of distances
		distances.append([distance, i])

	# sort the list
	distances = sorted(distances)

	# make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])

	# return most common target
	return Counter(targets).most_common(1)[0][0]

```
In the above code, we create an array of *distances* which we sort by increasing order. That way, we can grab the K nearest neighbors (first K distances), get their associated labels which we store in the *targets* array, and finally perform a majority vote using a *Counter*.

Putting it all together, we can define the function KNearestNeighbor, which loops over every test example and makes a prediction.

```python
def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	# train on the input data
	train(X_train, y_train)

	# loop over all observations
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i, :], k))
```

Let's go ahead and run our algorithm with the optimal K we found using cross-validation.

```python
# making our predictions 
predictions = []

kNearestNeighbor(X_train, y_train, X_test, predictions, 7)

# transform the list into an array
predictions = np.asarray(predictions)

# evaluating accuracy
accuracy = accuracy_score(y_test, predictions)
print('\nThe accuracy of our classifier is %d%%' % accuracy*100)
```
$$98\%$$ accuracy! We're as good as scikit-learn's algorithm, but probably less efficient. Let's try again with a value of $$K = 140$$. We get an `IndexError: list index out of range` error. In fact, K can't be arbitrarily large since we can't have more neighbors than the number of observations in the training data set. So let's fix our code to safeguard against such an error. Using `try, except` we can write the following code.

```python
def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	# check if k larger than n
	if k > len(X_train):
		raise ValueError
		
	# train on the input data
	train(X_train, y_train)

	# predict for each testing observation
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i, :], k))

# making our predictions 
predictions = []
try:
	kNearestNeighbor(X_train, y_train, X_test, predictions, 7)
	predictions = np.asarray(predictions)

	# evaluating accuracy
	accuracy = accuracy_score(y_test, predictions) * 100
	print('\nThe accuracy of OUR classifier is %d%%' % accuracy)

except ValueError:
	print('Can\'t have more neighbors than training samples!!')
```
That's it, we've just written our first machine learning algorithm from scratch!

## Pros and Cons of KNN

#### Pros

- simple to implement and understand
- zero to little training time 
- useful for off-the-bat analysis of data (maybe as a first step in understanding class distribution)
- useful for multiclass data sets

#### Cons

- computationally expensive at test time, which is undesirable in industry scenarios (compare this to ANN)
- training time can increase if we use Approximate Nearest Neighbor methods in high-dimension settings
- skewed class distributions will affect predictions

## Improvements

- weighted distances to prevent skewed classes dominating the predictions
- preprocess data: feature normalization
- dimensionality reduction: PCA or use of kernels
- Approximate Nearest Neighbor (KD tree, locality sensitive hashing)
- distance metric should be changed for different applications (i.e. hamming distance for text classification)

## Tutorial Summary

In this tutorial, we learned about the K-Nearest Neighbor algorithm, how it works and how it can be applied in a classification setting using scikit-learn's learning pipeline. We also implemented the algorithm in Python from scratch in such a way that we understand the inner-workings of the algorithm. Finally, we explored the pros and cons of KNN and the many improvements that can be made to adapt it to different project needs.

If you want to practice some more with the algorithm, try and run it on the __Breast Cancer Wisconsin__ dataset which you can find in the UC Irvine Machine Learning repository. You'll need to preprocess the data carefully this time. Do it once with scikit-learn's algorithm and a second time with our version of the code but try adding the weighted distance implementation. You can access the full code from this post [on my Github](https://github.com/kevinzakka/blog-code).
