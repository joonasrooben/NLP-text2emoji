# caption2emoji
**Aima Allik, Joonas Järve, Shumpei Morimoto**

### Objective
The plan is to create a NLP model that is capable to assign/predict emojis to given caption/tweet/text. 
To illustrate the idea, let's think of a typical predictive keyboard that suggest us the next words. This is very similar to our goal, the model will predict an emoji corresponding to the inserted text. The model itself will most probably be RNN or CNN. The idea is partly based on the article [Multimodal Emoji Prediction](https://arxiv.org/pdf/1803.02392.pdf) by Francesco Barbieri et al.

For researching the topic and writing a related work based on 3-5 papers we will use a selection from the following papers:
* https://arxiv.org/pdf/1803.02392.pdf
* https://core.ac.uk/reader/82522018
* https://arxiv.org/pdf/1707.04653.pdf
* https://arxiv.org/pdf/1609.08359.pdf

### Data
The main dataset will be 1M sentences with emoji from Twitter about 2017-Jan but the search is still in progress for new additional datasets. We are also deciding whether to use the emoji descriptions in training and/or testing. Our plan is not to learn emoji embeddings by ourselves but to use the ones already developed e.g emoji2vec [Learning Emoji Representations from their Description](https://arxiv.org/pdf/1609.08359.pdf) by Ben Eisner et al.

### Evaluation
As the subject is rather subjective we cannot really tell which is the gold label or if we are not assigning too big of an error to the wrong label, although they are very similar. Therefore, we could use a different measure. One way is to measure the predicted emoji similarity with the initial description of the emoji. The other way would be to evaluate the prediction which was in the users head. So, for that, we might need to carry out an extrinsic evaluation in the end to find out the model's actual performance.

