# caption2emoji
**Aima Allik, Joonas Järve, Shumpei Morimoto**

### Objective
The plan is to create a NLP model that is capable to assign/predict emojis to given caption/tweet/text. 
To illustrate the idea, let's think of a typical predictive keyboard that suggest us the next words. This is very similar to our goal, the model will predict an emoji corresponding to the inserted text. The model itself will most probably be RNN or CNN. The idea is partly based on the article [Multimodal Emoji Prediction](https://arxiv.org/pdf/1803.02392.pdf) by Francesco Barbieri et al.

For researching the topic and writing a related work based on 3-5 papers we will use a selection from the following papers:
* [Multimodal Emoji Prediction](https://arxiv.org/pdf/1803.02392.pdf) --JJ
* [Effective Emoticon Extractor for Behavior Analysis from Social Media](https://core.ac.uk/reader/82522018) 
* [A Semantics-Based Measure of Emoji Similarity](https://arxiv.org/pdf/1707.04653.pdf) --SM
* [Emoji Representations from their Description](https://arxiv.org/pdf/1609.08359.pdf) --AA
* [Emojitalianobot and EmojiWorldBot](http://ceur-ws.org/Vol-1749/paper37.pdf) --JJ
* [Emoji Prediction: Extensions and Benchmarking](https://arxiv.org/ftp/arxiv/papers/2007/2007.07389.pdf) --AA
* [Are Emojis Predictable?](https://arxiv.org/pdf/1702.07285.pdf) --JJ

### Data
The main dataset will be 1M sentences with emoji from Twitter about 2017-Jan but the search is still in progress for new additional datasets. We are also deciding whether to use the emoji descriptions in training and/or testing.

### Evaluation
In most of the articles, goodness is usually measured with traditional recall, precision, accuracy and F-score. In addition to the latter measures, as the target is rather subjective we cannot really tell which is the gold label or if we are not assigning too big of an error to the wrong label, although some emojis are very similar (e.g :smiley: and :smile:). Therefore, we could use a different measure. One way is to measure the predicted emoji similarity with the initial description of the emoji. The other way would be to evaluate the prediction which was in the users head. So, for that, we might need to carry out an extrinsic evaluation in the end to find out the model's actual performance.

### Related work
In this section we will give a slight overview of what has been done earlier and meantion the papers' similarities to our project.

* [Are Emojis Predictable?](https://arxiv.org/pdf/1702.07285.pdf)
After some additional research, we found a paper that very much does exactly what we are planning to do. In a way it was a good finding as it will provide us some more useful insights to the problem. The authors train a bidirectional LSTM to predict emojis based on tweets. They use word and character-based embeddings to represent tokens. The dataset consists of half a million tweets with only 20 different emojis. As a baseline model, they use *Bag of Words* and *skip-gram vector average* models. The best results were achieved with character-based embeddings and b-LSTM. In contrary to this article, we plan to try to concatenate the word- and character-based embeddings, use GRU instead and try with CNN as well. The amount to emojis taken to the project is still an open question.

* [Multimodal Emoji Prediction](https://arxiv.org/pdf/1803.02392.pdf)
This paper is an advancement of the previous paper by not only taking account the text but also picture. For such task, authors used Instagram posts. They used a smaller dataset but also pictures. They used the previous article as a baseline and using ResNet based model alongside with the *FastText* model. They showed that additional visual information provides slightly better results.

* [Emojitalianobot and EmojiWorldBot](http://ceur-ws.org/Vol-1749/paper37.pdf)
The authord of this paper have created a bot that translates textual language to emojis and vice versa. This is rather different from our goal but has brought attention to extrinsic evaluation means such as *crowdcourcing*. They also mention the annotations that provide labels for Unicode characters i.e [emojis](https://www.unicode.org/cldr/cldr-aux/charts/29/annotations/uralic.html) in our case.

* [A Semantics-Based Measure of Emoji Similarity](https://arxiv.org/pdf/1707.04653.pdf)
This paper analyzed the semantic similarity of emoji through embedding models using the word embedding model, which is called the emoji embedding model. Emojis were extracted into machine-readable emoji description(called sense) from EmojiNet. The author of this paper created a new dataset called EmoSim508 to evaluate the emoji embedding models' performance. EmoSim508 is a human-annotated semantic similarity score of 508 carefully selected emoji pairs. The emoji embedding model learned on sense labels correlate best with the created dataset's emoji similarity ranking. The paper also shows the result of the real-world use-case of the emoji embedding models, which is used in a sentiment analysis task.
