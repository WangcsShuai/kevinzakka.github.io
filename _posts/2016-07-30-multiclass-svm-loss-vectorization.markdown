---
layout: post
title: "Vectorizing the Multi-Class SVM Loss"
date: 2016-07-30
excerpt: "I'll explain how to vectorize the multiclass or hinge loss and provide example code."
tags: [vectorization, machine learning, classification, cs231n, visual, recognition]
comments: true
mathjax: true
---

In this blog post, we're going to vectorize the **Multiclass Support Vector Machine** or **Hinge** loss studied in Stanford's *Convolutional Neural Networks for Visual Recognition* (CS231n) online course. The goal is to introduce you to vectorization, an essential tool for Deep Learning and Machine Learning in general.

## Introduction

Let's start by motivating the concept of vectorization. In computer representation, an image is basically a 3D grid or array of pixels. In colored images, each pixel is a number which consists of 8 *bits*, meaning it can describe  $$2^8 = 256$$ colors, or levels of gray tones. 

The first dimension of an image represents its **width**, the second dimension represents its **height**, and the third dimension represents the **color channels** used.

<p align="center">
	<img src="/assets/pixels.png">
</p>

For example, in the landscape image above, the dimensions are 743 pixels wide, 364 pixels tall, and 3 color channels used. This means we have 743 x 364 x 3 = 811,356 pixels we're actually dealing with! 

You might be thinking that this is a piece of cake computation for any decent computer today, but remember, we're only dealing with 1 image here. If we have say 50,000 training images like the one above, we'd have, after stretching out each image into a 811,356 x 1 column vector, a 811,356 x 50,000 design matrix $X$.
  
It should come at no surprise then that calculations involving high-dimensional arrays are expensive both time and memory wise. Vectorizing comes in handy because it speeds up your code and usually looks cleaner which makes it less prone to bugs and errors.

In fact, consider the following block of code written first with `for` loops and then in vector form.

```python
```

I don't know about you, but I'd pick vectorization over the for loops any day.

## The Multiclass SVM Loss

I'll quickly summarize the key elements of the loss function to freshen your memory, but head over [here](http://cs231n.github.io/linear-classify/#svm) for all the well-explained details.

$$
L_{hinge} =  \frac{1}{N} \sum_i L_i
$$

where

$$
L_i = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta)
$$

Remember that $$s_j \\$$ represents the vector of class scores for a given image $$x_i$$. $$\Delta$$ is our threshold which can safely be set to $$1$$.

## Example Code

To illustrate our thought process, we'll be using small, random $$X$$, $$W$$, $$y$$ and $$b$$ matrices. Suppose we have: 

- 3 possible classes: 0, 1, 2 (or dog, cat, horse)
- 2 images consisting of 4 pixels each.

We thus have a matrix $$X$$ of 4 rows and 2 columns and a matrix $$W$$ of 3 rows (3 classes) and 4 columns.

Using the bias trick mentioned in the course, we introduce a new dimension of $$1$$'s to $$X$$ and add the bias column to the weight matrix $$W$$. This will simplify our work.

In conclusion, we have: 

- $$dim(X) = 5 \times 2$$ 
- $$dim(W) = 3 \times 5$$
- $$dim(y) = 2 \times 1$$


<p align="center">
	<img src="/assets/svmssoftmax.png">
</p>


Go ahead and copy paste the following code into Sublime:

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
# The correct class for the first image is 2 and for the second is 1.
y = np.array([2, 1])
```

## For Loop Version

This is the unvectorized function written by the course creators. I've altered the comments to make them uniform with our example code. Let's go ahead and dissect it to understand the big picture idea.

```python
def L_i(x, y, W):
	"""
		  Unvectorized version. Compute the multiclass svm loss for a single
		  example (x,y).
		  - x is a column vector representing an image with an appended bias
		  	dimension (5x1).
		  - y is an integer giving index of correct class (between 0 and 2)
		  - W is the weight matrix with the appended bias column (3x5)
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

## Fully-Vectorized Version

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
