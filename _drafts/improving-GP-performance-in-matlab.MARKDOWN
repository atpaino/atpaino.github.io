---
layout: post
title: Improving Genetic Programming Performance in Matlab
---
Intro
=====
I have been implementing a 
[Genetic Programming](http://en.wikipedia.org/wiki/Genetic_programming) (GP) 
algorithm for my research and have run into, and overcome, some severe 
performance issues. I am writing up the steps I took to achieve a usable level 
of performance so that it might be useful to others trying to write a GP 
algorithm (or just evaluate arithmetic expression trees) in Matlab.

The Problem
===========
In my research I am using GP to evolve image feature extraction algorithms with 
a structure similar to the 
[Histogram of Oriented Gradients](http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) (HOG) 
and [Local Binary Pattern](http://en.wikipedia.org/wiki/Local_binary_patterns) 
(LBP) algorithms. Each candidate solution in the algorithm is represented as 
an arithmetic expression tree that acts on a sub=cell of the image of 
predetermined size. To compute a feature vector for a given image, the 
expression represented by the tree must be evaluated for a sub=cell centered 
on every pixel in the image. Thus, an expression must be evaluated 128^2 times 
for an image with the dimensions 128x128. Additionally, GP usually calls for a 
population of at least 100 such candidate solutions, each of which must be 
evaluated on every image in a training set for at least 10 generations. As if 
this weren't enough, my implementation required me to evaluate the fitness 
function over a training set of 500 images.

For those keeping score, that means the algorithm will perform at least 
128\*128\*100\*10\*500 = **8.192 billion parse tree evaluations**! And that's 
before considering the possibility of having multiple features per candidate 
solution, which I have limited to 10. So in order for the algorithm to ever 
finish running, the evaluation of the expression better be very efficient.

Initial Approach: Recursive Evaluation
======================================
Evaluating a parse tree naturally lends itself to a recursive solution, so 
that was the first approach I took. However, when I first had this approach 
working, I realized it was way too slow to work. As in several orders of 
magnitude too slow. To get a handle on why, I ran Matlab's Profile tool on a 
candidate solution running on one image, which gave me the following output:
![Initial Performance Profile]({{ site.root }} /images/)

See that full blue bar next to "TODO: function call"? That means that almost 
all of the execution time was being spent in context switches due to the 
ridiculous number of recursive function calls.

Second Attempt: Building String Expressions and Using eval()
============================================================
After viewing the profile results from my initial approach, I begun thinking 
of ways to simplify the parse tree once before evaluating it. The solution I 
came up with was to build the arithmetic expression represented by a parse 
tree as a string, and then evaluate this expression using Matlab's `eval()` 
function.
![Second Attempt Performance Profile]({{ site.root }} /images/)

Getting Closer: Building Anonymous Functions using eval()
=========================================================
While the previous iteration drastically improved performance, it would still 
take weeks to run even a small version of the algorithm.
![Getting Closer Performance Profile]({{ site.root }} /images/)

Solution: Build a Single Multivariate Anonymous Function
========================================================
After reviewing the profile results from the previous run, I realized that 
most of the run time was being taken up my overtime involved with executing 
anonymous functions. Thus, in order to reduce the number of anonymous function 
calls I vector=ized the functions by having one anonymous function calculate 
all of the features for a parse tree. This greatly reduced the overhead of the 
anonymous functions by a factor of 10 (the limit on the number of features 
allowed per tree). 
![Final Performance Profile]({{ site.root }} /images/)

Conclusion
==========
If anyone has any other suggestions for improving performance in this 
application, please let me know. If you are interested in reading up on my 
research more, please see [this]({{ site.root }} /images/huhro.png ). Thanks for reading.
