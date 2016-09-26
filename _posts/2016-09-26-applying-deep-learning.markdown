---
layout: post
title: "Nuts and Bolts of Applying Deep Learning"
date: 2016-09-26
excerpt: "A comprehensive summary of Andrew Ng's talk at the Bay Area Deep Learning School (2016)"
tags: [deep learning, bias, variance, advice, end-to-end, machine learning]
comments: true
mathjax: true
---

This weekend was very hectic (catching up on courses and studying for a statistics quiz), but I managed to squeeze in some hours watching the [Bay Area Deep Learning School](http://www.bayareadlschool.org/) livestream on YouTube. For those of you wondering what that is, BADLS was a 2-day conference, hosted at Stanford University, and consisting of a back-to-back presentations on a variety of topics ranging from NLP, Computer Vision, Unsupervised Learning and Reinforcement Learning as well as tutorials on popular DL libraries in the industry today. 

There were some super interesting talks from leading experts in the field: [Hugo Larochelle](http://www.dmi.usherb.ca/~larocheh/index_en.html) from Twitter, [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/) from OpenAI, [Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/index.html) from the Université de Montreal, and [Andrew Ng](http://www.andrewng.org/) from Baidu to name a few. Of the plethora of presentations, there was one somewhat non-technical one given by Andrew that really piqued my interest. 

In this blog post, I'm gonna try and give a comprehensive summary of the main ideas of his talk. The goal is to pause a bit and take a look at the ongoing trends in Deep Learning thus far as well as gain some insight into applying DL in practice.

By the way, if you missed out on the livestreams, you can still view them at the following:

- [Day 1](https://www.youtube.com/watch?v=eyovmAtoUx0)
- [Day 2](https://www.youtube.com/watch?v=9dXiAecyJrY)


**Table of Contents**:

1. [Major Deep Learning Trends](#toc1)
2. [End-to-End Deep Learning](#toc2)
3. [Bias-Variance Tradeoff](#toc3)
4. [Human-level Performance](#toc4)
5. [Personal Advice](#toc5)

<a name='intro'></a>

### Major Deep Learning Trends
