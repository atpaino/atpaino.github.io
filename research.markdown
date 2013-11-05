---
layout: page
permalink: /research/index.html
title: Research
modified: 2013-11-01
---

Current Work
============
I'm currently working in the Computational Intelligence Research Lab at the 
University of Missouri. I'm working on using 
[Genetic Programming](http://en.wikipedia.org/wiki/Genetic_programming) (GP) 
to evolve image feature extraction algorithms that are an abstraction of the 
[Histogram of Oriented Gradients](http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) (HOG) 
and [Local Binary Pattern](http://en.wikipedia.org/wiki/Local_binary_patterns) 
algorithms. 

Past Work and Publications
=========

2013
----

### [Using evolutionary computation to optimize an SVM used in detecting buried objects in FLIR imagery](http://dx.doi.org/10.1117/12.2014774)

#### *Authors: Paino, Alex; Popescu, Mihail; Keller, James; Stone, Kevin*

__Abstract:__ In this paper we describe an approach for optimizing the parameters of a Support Vector Machine (SVM) as part of an algorithm used to detect buried objects in forward looking infrared (FLIR) imagery captured by a camera installed on a moving vehicle. The overall algorithm consists of a spot-finding procedure (to look for potential targets) followed by the extraction of several features from the neighborhood of each spot. The features include local binary pattern (LBP) and histogram of oriented gradients (HOG) as these are good at detecting texture classes. Finally, we project and sum each hit into UTM space along with its confidence value (obtained from the SVM), producing a confidence map for ROC analysis. In this work, we use an Evolutionary Computation Algorithm (ECA) to optimize various parameters involved in the system, such as the combination of features used, parameters on the Canny edge detector, the SVM kernel, and various HOG and LBP parameters. To validate our approach, we compare results obtained from an SVM using parameters obtained through our ECA technique with those previously selected by hand through several iterations of "guess and check".
\[[pdf](/pdfs/SPIE2013_8709-47ECAmanuscriptPainoFINAL.pdf)\]

2012
----

### [Detection of buried objects in FLIR imaging using mathematical morphology and SVM](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6291520&isnumber=6291498)

#### *Authors: Popescu, Mihail; Paino, Alex; Stone, Kevin; Keller, James*

__Abstract:__ In this paper we describe a method for detecting buried objects of interest using a forward looking infrared camera (FLIR) installed on a moving vehicle. Infrared (IR) detection of buried targets is based on the thermal gradient between the object and the surrounding soil. The processing of FILR images consists in a spot-finding procedure that includes edge detection, opening and closing. Each spot is then described using texture features such as histogram of gradients (HOG) and local binary patterns (LBP) and assigned a target confidence using a support vector machine (SVM) classifier. Next, each spot together with its confidence is projected and summed in the UTM space. To validate our approach, we present results obtained on 6 one mile long runs recorded with a long wave IR (LWIR) camera installed on a moving vehicle.
\[[pdf](/pdfs/flir_cisda2012_last.pdf)\]
