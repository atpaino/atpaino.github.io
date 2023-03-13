---
layout: post
title: Deep Text Corrector
modified: 2017-01-03
excerpt: In this post I describe Deep Text Corrector, a system built on deep learning that is capable of correcting small grammatical errors in short, message-like text.
---

[**Code**](https://github.com/atpaino/deep-text-corrector)
|

While context-sensitive spell-check systems (such as [AutoCorrect](https://en.wikipedia.org/wiki/Autocorrection)) are able to automatically correct a large number of input errors in instant messaging, email, and SMS messages, they are unable to correct even simple grammatical errors. 
For example, the message "I'm going to store" would be unaffected by typical autocorrection systems, when the user most likely intendend to communicate "I'm going to _the_ store". 

Inspired by recent advancements in NLP driven by deep learning (such as those in Neural Machine Translation by [Bahdanau et al., 2014](http://arxiv.org/abs/1409.0473)), I decided to try training a neural network to solve this problem. 
Specifically, I set out to construct sequence-to-sequence models capable of processing a sample of conversational written English and generating a corrected version of that sample. 
In this post I'll describe how I created this "Deep Text Corrector" system and present some encouraging initial results. 

All code is available on GitHub [here](https://github.com/atpaino/deep-text-corrector).html).

## Correcting Grammatical Errors with Deep Learning
The basic idea behind this project is that we can generate large training datasets for the task of grammar correction by starting with grammatically correct samples and introducing small errors to produce input-output pairs.
The details of how we construct these datasets, train models using them, and produce predictions for this task are described below.

### Datasets
To create a dataset for training Deep Text Corrector models, I started with a large collection of mostly grammatically correct samples of conversational written English. 
The primary dataset considered in this project is the [Cornell Movie-Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), which contains over 300k lines from movie scripts.
This was the largest collection of conversational written English I could find that was (mostly) grammatically correct. 

Given a sample of text like this, the next step is to generate input-output pairs to be used during training. 
This is done by:

1. Drawing a sample sentence from the dataset.
2. Setting the input sequence to this sentence after randomly applying certain perturbations.
3. Setting the output sequence to the unperturbed sentence.

where the perturbations applied in step (2) are intended to introduce small grammatical errors which we would like the model to learn to correct. 
Thus far, these perturbations have been limited to:

- the subtraction of articles (a, an, the)
- the subtraction of the second part of a verb contraction (e.g. "'ve", "'ll", "'s", "'m")
- the replacement of a few common homophones with one of their counterparts (e.g. replacing "their" with "there", "then" with "than")

For example, given the sample sentence:

    And who was the enemy?

the following input-output pair could be generated:

    ("And who was enemy ?", "And who was the enemy ?")

The rates with which these perturbations are introduced are loosely based on figures taken from the [CoNLL 2014 Shared Task on Grammatical Error Correction](http://www.aclweb.org/anthology/W14-1701.pdf). 
In this project, each perturbation is randomly applied in 25% of cases where it could potentially be applied.

### Training
In order to artificially increase the dataset when training a sequence-to-sequence model, I performed the sampling strategy described above multiple times over the Movie-Dialogs Corpus to arrive at a dataset 2-3x the size of the original corups. 
Given this augmented dataset, training proceeded in a very similar manner to [TensorFlow's sequence-to-sequence tutorial](https://www.tensorflow.org/tutorials/seq2seq/). 
That is, I trained a sequence-to-sequence model consisting of LSTM encoders and decoders bridged via an attention mechanism, as described in [Bahdanau et al., 2014](http://arxiv.org/abs/1409.0473). 

### Decoding

Instead of using the most probable decoding according to the sequence-to-sequence model, this project takes advantage of the unique structure of the problem to impose the prior that all tokens in a decoded sequence should either exist in the input sequence or belong to a set of "corrective" tokens. 
The "corrective" token set is constructed during training and contains all tokens seen in the target, but not the source, for at least one sample in the training set. 
The intuition here is that the errors seen during training involve the misuse of a relatively small vocabulary of common words (e.g. "the", "an", "their") and that the model should only be allowed to perform corrections in this domain.

This prior is carried out through a modification to the decoding loop in TensorFlow's seq2seq model in addition to a post-processing step that resolves out-of-vocabulary (OOV) tokens:

**Biased Decoding**

To restrict the decoding such that it only ever chooses tokens from the input sequence or corrective token set, this project applies a binary mask to the model's logits prior to extracting the prediction to be fed into the next time step. 

This is done by constructing a mask:
{% highlight python %}
mask[i] == 1.0 if i in (input or corrective_tokens) else 0.0 
{% endhighlight %}

and then using it during decoding in the following manner:
{% highlight python %}
token_probs = tf.softmax(logits)
biased_token_probs = tf.mul(token_probs, mask)
decoded_token = math_ops.argmax(biased_token_probs, 1)
{% endhighlight %}

Since this mask is applied to the result of a softmax transformation (which guarantees all outputs are positive), we can be sure that only input or corrective tokens are ever selected.

Note that this logic is not used during training, as this would only serve to hide potentially useful signal from the model.

**Handling OOV Tokens**

Since the decoding bias described above is applied within the truncated vocabulary used by the model, we will still see the unknown token in its output for any OOV tokens. 
The more generic problem of resolving these OOV tokens is non-trivial (e.g. see [Addressing the Rare Word Problem in NMT](https://arxiv.org/pdf/1410.8206v4.pdf)), but in this project we can again take advantage of the unique structure of the problem to create a fairly straightforward OOV token resolution scheme. 

Specifically, if we assume the sequence of OOV tokens in the input is equal to the sequence of OOV tokens in the output sequence, then we can trivially assign the appropriate token to each "unknown" token encountered in the decoding. 

For example, in the following scenario:

    Input sequence: "Alex went to store"
    Target sequence: "Alex went to the store"
    Decoding from model: "UNK went to the store"

this logic would replace the `UNK` token in the decoding with `Alex`.

Empirically, and intuitively, this appears to be an appropriate assumption, as the relatively simple class of errors these models are being trained to address should never include mistakes that warrant the insertion or removal of a rare token.

## Experiments and Results

Below are some anecdotal and aggregate results from experiments using the Deep Text Corrector model with the [Cornell Movie-Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). 
The dataset consists of 304,713 lines from movie scripts, of which 243,768 lines were used to train the model and 30,474 lines each were used for the validation and testing sets. 
For the training set, 2 samples were drawn per line in the corpus, as described above. 
The sets were selected such that no lines from the same movie were present in both the training and testing sets.

The model being evaluated below is a sequence-to-sequence model, with attention, where the encoder and decoder were both 2-layer, 512 hidden unit LSTMs. 
The model was trained with a vocabulary consisting of the 2,000 most common words seen in the training set (note that we can use a small vocabulary here due to our OOV resolution strategy). 
A bucketing scheme similar to that in [Bahdanau et al., 2014](http://arxiv.org/abs/1409.0473) is used, resulting in 4 models for input-output pairs of sizes smaller than 10, 15, 20, and 40. 

### Aggregate Performance
Below are reported the corpus BLEU scores (as computed by [NLTK](http://www.nltk.org/api/nltk.translate.html)) and accuracy numbers over the test dataset for both the trained model and a baseline. 
The baseline used here is simply the identity function, which assumes no errors exist in the input; the motivation for this is to test whether the introduction of the trained model could add value to an existing system with no grammar-correction system in place. 

Encouragingly, **the trained model outperforms this baseline** for all bucket sizes in terms of accuracy, and outperforms all but one in terms of BLEU score. 
This tells us that applying the Deep Text Corrector model to a potentially errant writing sample would, on average, result in a more grammatically correct writing sample. 
Anyone who tends to make errors similar to those the model has been trained on could therefore benefit from passing their messages through this model.


|Bucket (seq length)|Baseline BLEU|Model BLEU|Baseline Accuracy|Model Accuracy|
|---|---|---|---|---|
|Bucket 1 (10)|0.8341|0.8516|0.9083|0.9384|
|Bucket 2 (15)|0.8850|0.8860|0.8156|0.8491|
|Bucket 3 (20)|0.8876|0.8880|0.7291|0.7817|
|Bucket 4 (40)|0.9099|0.9045|0.6073|0.6425|
{:.mbtablestyle}  

<br />

### Examples

In addition to the encouraging aggregate performance of this model, we can see that its is capable of generalizing beyond the specific language styles present in the Movie-Dialogs corpus by testing it on a few fabricated, grammatically incorrect sentences. 
Below are a few examples.

**Decoding a sentence with a missing article:**

{% highlight python %}
In [31]: decode("Kvothe went to market")
Out[31]: 'Kvothe went to the market'
{% endhighlight %}

**Decoding a sentence with then/than confusion:**

{% highlight python %}
In [30]: decode("the Cardinals did better then the Cubs in the offseason")
Out[30]: 'the Cardinals did better than the Cubs in the offseason'
{% endhighlight %}

Note that in addition to correcting the grammatical errors, the system is able to handle OOV tokens without issue.

## Future Work
While these initial results are encouraging, there is still a lot of room for improvement. 
The biggest thing holding the project back is the lack of a large dataset -- the 300k samples in the Cornell Movie Dialogs dataset is tiny by modern deep learning standards. 
Unfortunately, I am not aware of any publicly available dataset of (mostly) grammatically correct English. 
A close proxy could be comments in a "higher quality" online forum, such as Hacker News or certain subreddits. 
I may try this next. 

Given a larger dataset, I would also like to try to introduce many different kinds of errors into the training samples. 
The perturbations used thus far are limited to fairly simple grammatical mistakes; it would be very interesting to see if the model could learn to correct somewhat more subtle mistakes, such as subject-verb agreement. 

On the application front, I could see this system eventually being accessible via a "correction" API that could be leveraged in a variety of messaging applications. 

[Comments on HN](https://news.ycombinator.com/item?id=13350972)
