---
layout: post
title: "Deriving the Gradient for the Backward Pass of Batch Normalization"
date: 2016-09-14
excerpt: "Deriving an expression for the gradient in Batch Normalization."
tags: [batch normalization, gradient, chain rule, cs231n]
comments: true
mathjax: true
---

I recently sat down to work on assignment 2 of Stanford's [CS231n](http://cs231n.github.io/assignments2016/assignment2/). It's lengthy and definitely a step up from the first assignment, but the insight you gain is tremendous. 

Anyway, at one point in the assignment, we were tasked with implementing a Batch Normalization layer in our fully-connected net which required writing a forward and backward pass.

The forward pass is relatively simple since it only requires standardizing the input features (zero mean and unit standard deviation). The backwards pass, on the other hand, is a bit more involved. It can be done in 2 different ways:

- **staged computation**: we can break up the function into several parts, derive local gradients for them, and finally multiply them with the chain rule.
- **gradient derivation**: basically, you have to do a "pen and paper" derivation of the gradient with respect to the inputs.

It turns out that second option is faster, albeit nastier and after struggling for a few hours, I finally got it to work. This post is mainly a clear summary of the derivation along with my thought process, and I hope it can provide others with the insight and intuition of the chain rule.

By the way, I have summarized the [paper](https://arxiv.org/abs/1502.03167) and accompanied it with a small implementation which you can view on my [Github](https://github.com/kevinzakka/research-paper-notes). Without further ado, let's jump right in.

### Notation

Let's start with some notation.

- **BN** will stand for Batch Norm.
- $$f$$ represents a layer upwards of the BN one.
- $$y$$ is the linear transformation which scales $x$ by $\gamma$ and adds $\beta$.
- $$\hat{x}$$ is the normalized inputs.
- $$\mu$$ is the batch mean.
- $$\sigma^2$$ is the batch variance.

The below table shows you the inputs to each function and will help with the future derivation.   

<center>

| $$f(y)$$  | $$y(\hat{x}, \gamma, \beta)$$  | $$\hat{x}(x, \mu, \sigma^2)$$  |  $$\sigma^2(x, \mu)$$ | $$\mu(x)$$  |
|---|---|---|---|---|

</center>

**Goal**: Find the partial derivatives with respect to the inputs, that is $$\frac{\partial f}{\partial \gamma}$$, $$\frac{\partial f}{\partial \beta}$$ and $$\frac{\partial f}{\partial x}$$.

The methodology we will adopt will be to derive the gradient with respect to the centered inputs $$\hat{x}$$ (which requires deriving the gradient w.r.t $$\mu$$ and $$\sigma^2$$) and then use those to derive one for $$x$$.

### Chain Rule Primer

Suppose we're given a function $$u(x, y)$$ where $$x(r, t)$$ and $$y(r, t)$$. Then to determine the value of $$\frac{\partial u}{\partial r}$$ and $$\frac{\partial u}{\partial t}$$ we need to use the chain rule which says that:

$$\frac{\partial u}{\partial r} = \frac{\partial u}{\partial x} \frac{\partial x}{\partial r} + \frac{\partial u}{\partial y} \frac{\partial y}{\partial r}$$

That's basically all there is to it. Using this simple concept can help us solve our problem. We just have to be clear and precise when using it and not get lost with the intermediate variables.

### Partial Derivatives

Here's the gist of BN taken from the paper.

<p align="center">
 <img src="/assets/alg1.png" width="380">
</p>

We're gonna start by traversing the table from left to right. At each step we derive the gradient with respect to the inputs in the cell.

$$ \frac{\partial f}{\partial y_i} $$

This derivative is already computed for us and is represented by the variable `dout` in the assignment. Moving on to cell 2. We note that $$y$$ is a function of three variables, so let's compute the gradient with respect to those 3.

$$ \frac{\partial f}{\partial \gamma} $$
We need to use the chain rule here. 
$$
\begin{eqnarray}
\frac{\partial f}{\partial \gamma} &=& \frac{\partial f}{\partial y_i} \cdot \frac{\partial y_i}{\partial \gamma} \qquad \\
\frac{\partial f}{\partial \gamma} &=& \sum\limits_{i=1}^m \frac{\partial f}{\partial y_i} \cdot \hat{x}_i
\end{eqnarray}
$$

Notice that we sum from $$1 \rightarrow m$$ because we're working with batches! Moving on to $$\beta$$ we compute the gradient as follows:

$$
\begin{eqnarray}
\frac{\partial f}{\partial \beta} &=& \frac{\partial f}{\partial y_i} \cdot \frac{\partial y_i}{\partial \beta} \qquad \\
\frac{\partial f}{\partial \beta} &=& \sum\limits_{i=1}^m \frac{\partial f}{\partial y_i}
\end{eqnarray}
$$

Up to now, things are relatively simple and we've already done 2/3 of the variables. We can't compute the gradient w.r.t $$x$$ just yet though. Moving on to $$\hat{x}$$:

$$
\begin{eqnarray}
\frac{\partial f}{\partial \hat{x}} &=& \frac{\partial f}{\partial y_i} \cdot \frac{\partial y_i}{\partial \hat{x}} \qquad \\
\frac{\partial f}{\partial \hat{x}} &=& \frac{\partial f}{\partial y_i} \cdot \gamma
\end{eqnarray}
$$
