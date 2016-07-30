---
layout: post
title: "Vectorizing the Multi-Class SVM Loss"
date: 2016-07-30
excerpt: "I'll explain how to vectorize the multiclass SVM or hinge loss and provide example code."
tags: [vectorization, machine learning, classification, cs231n, visual recognition]
comments: true
mathjax: true
---

In this blog post, we're going to vectorize the **Multiclass Support Vector Machine** or **Hinge** loss studied in Stanford's *Convolutional Neural Networks for Visual Recognition* (CS231n) online course. The goal is to introduce you to vectorization, an essential tool for Deep Learning and Machine Learning in general.

For the full code that appears on this page, visit my [Github Repository](https://github.com/kevinzakka/blog-code/).

## Table of Contents

1. [Motivation](#intro)
2. [The Multiclass SVM Loss](#loss)
3. [Example Code](#code)
4. [For Loop Version](#loop)
5. [Fully-Vectorized Version](#vectorized)
6. [Speed-Up](#comp) 
7. [Summary](#summary)

<a name='intro'></a>

## Motivation

Let's start by motivating the concept of vectorization. In computer representation, an image is basically a 3D grid or array of pixels. In colored images, each pixel is a number which consists of 8 *bits*, meaning it can describe  $$2^8 = 256$$ colors, or levels of gray tones. 

The first dimension of an image represents its **width**, the second dimension represents its **height**, and the third dimension represents the number of **color channels** used.

<p align="center">
	<img src="/assets/pixels.png">
</p>

For example, in the landscape image above, the dimensions are 743 pixels wide, 364 pixels tall, and 3 color channels (RGB) used. This means we have 743 x 364 x 3 = 811,356 pixels we're actually dealing with! 

You might be thinking that this is a piece of cake computation for any decent computer today, but remember, we're only dealing with 1 image here. If we have say 50,000 training images like the one above, we'd have, after stretching out each image into a 811,356 x 1 column vector, a 811,356 x 50,000 design matrix $$X$$.
  
It should come at no surprise then that calculations involving huge, high-dimensional arrays are expensive both time and memory wise. Vectorizing those calculations comes in handy because not only does it speed up your code but it usually looks much cleaner making it easier to read and debug later on.

Consider the following vector addition written first with `for` loops and then in `vector` form.

```python
# ===================== unvectorized code =====================
x = np.array([1, 2, 3])
y = np.array([2, 3, 4])

# initialize result vector z
z = np.zeros(shape=(3,))

for i in range(x.shape[0]):
	z[i] = x[i] + y[i]

# ====================== vectorized code =======================
x = np.array([1, 2, 3])
y = np.array([2, 3, 4])

z = x + y
```

As you can see, the vectorized version is shorter and aesthetically superior. The addition operation is done on all three components of `x` and `y` at the same time, meaning we don't need a `for` loop! 

Hence, rather than sequentially dealing with each component of the vector at a time, vectorized code enables us to abstract to a single dimension and treat each vector as a unit. Note that you should never write something like the unvectorized version - numpy commands are there to specifically avoid that - it is just for illustration purposes.

> Vectorization frees us from having to loop through a vector one component at a time.

<a name='loss'></a>

## The Multiclass SVM Loss

I'll quickly summarize the key elements of the loss function to refresh your memory, but head over to [Linear classification: Support Vector Machine, Softmax](http://cs231n.github.io/linear-classify/#svm) under Module 1 of the course for all the well-explained details.

$$
L_{hinge} =  \frac{1}{N} \sum_i L_i
$$

where

$$
L_i = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta)
$$

Remember that $$s_j \\$$ represents the vector of class scores for a given image $$x_i$$. $$\Delta$$ is our threshold which can safely be set to $$1$$.

<a name='code'></a>

## Example Code

As the famous adage says, **code is worth a thousand words** (or was it images 😉) so let's go ahead and create some example $$X$$, $$W$$, $$y$$ and $$b$$ matrices that we can play around with. Suppose we have: 

- 2 images consisting of 4 pixels each.
- 3 possible classes: dog, cat, and horse

We thus have a matrix $$X$$, 4 rows long and 2 columns wide and a matrix $$W$$, 3 rows long (3 classes) and 4 columns wide.

Using the **bias trick** mentioned in the course, we introduce a new dimension of $$1$$'s to $$X$$ and add the bias column to the weight matrix $$W$$ (this will make it so we don't have to deal with an extra bias matrix). In conclusion, we have: 

- dim(X) = 5 x 2
- dim(W) = 3 x 5
- dim(y) = 2 x 1

Go ahead and copy paste the following code into your favorite editor, in my case Sublime Text:

```python
# loading packages
import numpy as np

# dim(X) = 5x2 = 2 training observations consisting of 5 pixel values each
X = np.array([
	[-15, 2], 
	[22, 11], 
	[-44, -35], 
	[56, 12], 
	[1.0, 1.0]
])
# dim(W) = 3x5 
W = np.array([
	[0.01, -0.05, 0.1, 0.05, 0.0], 
	[0.7, 0.2, 0.05, 0.16, 0.2], 
	[0.0, -0.45, -0.2, 0.03, -0.3]
])
# remember that y holds the integers corresponding to the correct classes
y = np.array([2, 1])
```

<a name='loop'></a>

## For-loop Version

This is the unvectorized function written by the course creators. I've altered the comments to make them uniform with our example code. Let's go ahead and dissect it to understand the big picture idea.

```python
def L_i(x, y, W):
	"""
	Unvectorized version. Compute the multiclass SVM loss for a single
	example (x,y).
	- x is a column vector representing an image with an appended bias
	dimension (5x1).
	- y is an integer giving index of correct class (between 0 and 2).
	- W is the weight matrix with the appended bias column (3x5).
	"""
	delta = 1.0 			
	scores = W.dot(x)
	correct_class_score = scores[y]
	D = W.shape[0] 			# number of classes, e.g. 3
	loss_i = 0.0
	
	for j in range(D): 		# iterate over all wrong classes
		if j == y:
	  		continue
		# accumulate loss for the i-th example
		loss_i += max(0, scores[j] - correct_class_score + delta)

	return loss_i
```

The above function corresponds to the loss associated to a single image $$x_i$$. Remember that we need to sum over the incorrect class scores, and calculate the associated margin $$\max{(0, s_j - s_{y_i} + \Delta )} \\$$.

Here's a picture that describes what the loss function is doing.

<p align="center">
	<img src="/assets/dot_product.png">
</p>

Thus we start by calculating $$s_j$$. This is done by computing the dot product between the weight matrix $$W$$ and our image $$x$$, or `scores = W.dot(x)`. Next, we need to determine the score of the correct class. We have $$y$$, so let's just index into the `scores` matrix we just calculated and grab $$s_{y_i}$$. Now, all we need to do is loop over the incorrect classes. `if j == y` makes sure we skip the loss accumulation operation in the loop.

Our final result is `loss_i` which corresponds to the loss for just 1 image. We need to do this for both images in our dataset and compute the average loss. The code would go something like this:

```python
N = X.shape[1]
# initialize loss
losses = 0.0

# loop over each image
for i in range(N):
	# accumulate loss
	losses += L_i(X[:, i], y[i], W)
	
totalLoss = losses / N
print('L = {}'.format(totalLoss))
```

<a name='vectorized'></a>

## Fully-Vectorized Version
For the sake of clarity, let's summarize the steps needed to compute the SVM loss:

1. Compute scores
2. Compute margins while being careful to exclude correct classes
3. Sum the margins to get loss
4. Average the loss and output result

We're gonna do exactly the same thing with the vectorized version. Let's start with the first step. Instead of dealing with a single image now, we'll be dealing directly with $$X$$, where each image is stored as a column.

```python
def L(X, y, W):
	"""
	Fully-vectorized implementation.
	- X holds all the training examples as columns.
	- y is array of integers specifying correct class.
  	- W holds the weights.
	"""
	# grab number of images
	N = X.shape[1]
	# set desired threshold
	delta = 1.0

	# scores holds the score for each image as columns
	scores = W.dot(X)
```

This is super easy. All we did was replace `x` with `X`. The resulting matrix `scores` stores the scores for each image as columns. Let's perform a sanity check.

```python
x = X[:, 0]
scores = W.dot(x)

# ==== Console Print-Out ===
>>> scores
array([-2.85,  0.86,  0.28])
# ==========================

scores = W.dot(X)

# ==== Console Print-Out ===
>>> scores
array([[-2.85, -3.43],
       [ 0.86,  3.97],
       [ 0.28,  2.11]])
```
All good. The next step is to compute the margins. This requires 4 things: grabbing the correct class **for each image**, subtracting that score from the incorrect scores, subsequently adding delta, and finally taking the **max()** with 0. 

The difficulty here lies in grabbing the correct class scores. The rest is a piece of cake. To solve this little crux, we're gonna use a special type of numpy array indexing. 

```python
# grab scores of correct classes
correct_classes = scores[y, np.arange(N)]
```

This type of indexing lets us select one element from each column of `scores` using the indices in our vector `y`. Super cool right?

Now we need to subtract, add and squash the scores. Doing this requires that we vectorize a squash function that we will write using numpy's `vectorize` method. Vectorizing in numpy makes it so that a function can be applied element-wise on a matrix.

```python
def squash(x, delta):
	return np.maximum(0, x + delta)
	
# vectorize squash function
vectorizedSquash = np.vectorize(squash, otypes=[np.float])

# compute margins element-wise
margins = vectorizedSquash(scores - correct_classes, delta)
```

Almost done! Remember that we need to ignore the losses on the correct classes. So let's reuse the array indexing and set those to 0 with `margins[y, np.arange(2)] = 0`. All we have left is to sum the losses and average them out.

```python
# compute loss column-wise
losses = np.sum(margins, axis=0)
	
# return average loss
return (np.sum(losses) / N)
```

Give yourself a pat on the back! We've done it. We've fully vectorized the code of the multiclass SVM loss. The bird's eye view of the function is posted below:

```python
def squash(x, delta):
	return np.maximum(0, x + delta)
	
def L(X, y, W):
	"""
	Fully-vectorized implementation
	- X holds all the training examples as columns
	- y is array of integers specifying correct class
 	- W holds the weights
	"""
	# grab number of images
	N = X.shape[1]
	# set desired threshold
	delta = 1.0
	
	# scores holds the score for each image as columns
	scores = W.dot(X)
	
	# grab scores of correct classes
	correct_classes = scores[y, np.arange(N)]
	
	# vectorize squash function
	vectorizedSquash = np.vectorize(squash, otypes=[np.float])
	
	# compute margins element-wise
	margins = vectorizedSquash(scores - correct_classes, delta)
	
	# ignore the y-th position and only consider margin on max wrong class
	margins[y, np.arange(2)] = 0
	
	# compute loss column-wise
	losses = np.sum(margins, axis=0)
	# return average loss
	return (np.sum(losses) / N)
```

<a name='comp'></a>

## Speed-Up
Let's do some quick time comparisons to see if we actually get any speed improvements.

Adding to the above code:

```python
import time

start_time = time.time()
print('loss: {} '.format(L(X, y, W))
time1 = time.time() - start_time
print('Vectorized takes: {}'.format(time1))

start_time = time.time()
for i in range(X.shape[1]):
	losses += L_i(X[:, i], y[i], W)
print('loss: '.format(losses / X.shape[1]))
time2 = time.time() - start_time
print('Non-vectorized takes: {}'.format(time2))

print( ((time1 - time2) / time2) * 100 )

# ============ Console Print-Out ===========
loss:  0.79
Vectorized takes: 0.0009570121765136719
loss:  0.79
Non-vectorized takes: 6.890296936035156e-05
560.9756097560976
```
A whooping 561% improvement! Give yourself another pat on the back.

<a name='summary'></a>

## Summary

In this blog post, we learned about the advantages of vectorization specifically in visual recognition applications where dealing with very high-dimensional arrays is very common. We went from a double for-loop implementation to a fully vectorized one with some example code to illustrate each step.

As mentioned previously, this post is meant to help people taking CS231n. You can:

- Visit the course syllabus: click [here](http://cs231n.github.io/)
- Read the section on SVM and Softmax Loss - click [here](http://cs231n.github.io/linear-classify/)
