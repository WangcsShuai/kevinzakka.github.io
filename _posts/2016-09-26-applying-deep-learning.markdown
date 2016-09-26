---
layout: post
title: "Nuts and Bolts of Applying Deep Learning"
date: 2016-09-26
excerpt: "A comprehensive summary of Andrew Ng's talk at the 2016 Bay Area Deep Learning School"
tags: [deep learning, bias, variance, advice, end-to-end, machine learning]
comments: true
mathjax: true
---

This weekend was very hectic (catching up on courses and studying for a statistics quiz), but I managed to squeeze in some time to watch the [Bay Area Deep Learning School](http://www.bayareadlschool.org/) livestream on YouTube. For those of you wondering what that is, BADLS is a 2-day conference hosted at Stanford University, and consisting of back-to-back presentations on a variety of topics ranging from NLP, Computer Vision, Unsupervised Learning and Reinforcement Learning. Additionally, top DL software libraries were presented such as Torch, Theano and Tensorflow. 

There were some super interesting talks from leading experts in the field: [Hugo Larochelle](http://www.dmi.usherb.ca/~larocheh/index_en.html) from Twitter, [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/) from OpenAI, [Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/index.html) from the Université de Montreal, and [Andrew Ng](http://www.andrewng.org/) from Baidu to name a few. Of the plethora of presentations, there was one somewhat non-technical one given by Andrew that really piqued my interest. 

In this blog post, I'm gonna try and give a comprehensive summary of the main ideas outlined in his talk. The goal is to pause a bit and take a look at the ongoing trends in Deep Learning thus far, as well as gain some insight into applying DL in practice.

By the way, if you missed out on the livestreams, you can still view them at the following: [Day 1](https://www.youtube.com/watch?v=eyovmAtoUx0), [Day 2](https://www.youtube.com/watch?v=9dXiAecyJrY).


**Table of Contents**:

1. [Major Deep Learning Trends](#toc1)
2. [End-to-End Deep Learning](#toc2)
3. [Bias-Variance Tradeoff](#toc3)
4. [Human-level Performance](#toc4)
5. [Personal Advice](#toc5)

<a name='toc1'></a>

### Major Deep Learning Trends

- **Why do DL algorithms work so well?**

According to Ng, with the rise of the Internet, Mobile and IOT era, the amount of data accessible to us has greatly increased. This correlates directly to a boost in the performance of neural network models, especially the larger ones which have the capacity to absorb all this data.

<p align="center">
 <img src="/assets/app_dl/perf_vs_data.png" width="450">
</p>

However, in the small data regime (left-hand side of the x-axis), the relative ordering of the algorithms is not that well defined and really depends on who is more motivated to engineer their features better, or refine and tune the hyperparameters of their model. 

Thus this trend is more prevalent in the big data realm where hand engineering effectively gets replaced by end-to-end approaches and bigger neural nets combined with a lot of data tend to outperform all other models.

- **Machine Learning and HPC team** 

The rise of big data has put pressure on companies in the sense that a *Computer Systems* team is now required. This is because some of the HPC (high-performance computing) applications require highly specialized knowledge and it is difficult to find researchers and engineers with sufficient knowledge in both fields. Knowledge and cooperation from both teams is then the key to boosting performance in AI companies.

- **Categorizing DL models**

Work in DL can be categorized in the following 4 buckets.

<p align="center">
 <img src="/assets/app_dl/bucket.svg" width="350">
</p>

Most of the value in the industry today is driven by the models in the orange rectangle (innovation and monetization mostly) but Andrew believes that **unsupervised deep learning** is a super-exciting field that has lots of potential for the future.

<a name='toc2'></a>

### The rise of End-to-End DL

A major improvement in the end-to-end approach has been the fact that outputs are becoming more and more complicated. For example, rather than just outputting a simple class score such as 0 or 1, algorithms are starting to generate richer outputs: images like in the case of GAN's, full captions with RNN's and most recently, audio like in DeepMind's WaveNet.

**So what exactly does end-to-end training mean?**

Essentially, it means that AI practitioners are shying away from intermediate representations and going directly from one end (raw input) to the other end (output) Here's an example from speech recognition.

<p align="center">
 <img src="/assets/app_dl/end-to-end.svg" width="340">
</p>

**Are there any disadvantages to this approach?**

End-to-end approaches are data hungry meaning they only perform well when provided with a huge dataset of labelled examples. In practice, not all applications have the luxury of large labelled datasets so other approaches which allow hand-engineered information and field expertise to be added into the model have gained the upper hand. As an example, in a self-driving car setting, going directly from the raw image to the steering direction is pretty difficult. Rather, many features such as trajectory and pedestrian location are calculated first as intermediate steps.


The main take-away from this section is that we should always be cautious of end-to-end approaches in applications where huge data is hard to come by. 

<a name='toc3'></a>

### Bias-Variance Tradeoff
