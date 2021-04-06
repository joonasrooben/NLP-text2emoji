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
As the subject is rather subjective we cannot really tell which is the gold label or if we are not assigning too big of an error to the wrong label, although some emojis are very similar (e.g :smiley: and :smile:). Therefore, we could use a different measure. One way is to measure the predicted emoji similarity with the initial description of the emoji. The other way would be to evaluate the prediction which was in the users head. So, for that, we might need to carry out an extrinsic evaluation in the end to find out the model's actual performance.

