# caption2emoji
**Aima Allik, Joonas Järve, Shumpei Morimoto**

### Objective
The plan is to create a NLP model that is capable to assign/predict emojis to given caption/tweet/text. 
To illustrate the idea, let's think of a typical predictive keyboard that suggest us the next words. This is very similar to our goal, the model will predict an emoji corresponding to the inserted text. The model itself will most probably be RNN on CNN. The idea is partly based on the article [Multimodal Emoji Prediction](https://arxiv.org/pdf/1803.02392.pdf) by Francesco Barbieri et al.

### Data
The main dataset will most probably be 1M sentences with emoji from Twitter about 2017-Jan but the search is still in progress for new datasets. We are also deciding whether to use the emoji descriptions also in training. Our plan is not to learn emoji embeddings by ourselves but to use the ones already developed e.g emoji2vec [Learning Emoji Representations from their Description](https://arxiv.org/pdf/1609.08359.pdf) by Ben Eisner et al.

### Evaluation
As the subject is rather fuzzy we cannot really tell which is the gold label or if we are not assigning to big of an error to the worng label, altough they are very similar. Therefore, we could use a different measure. One way is to measure the predicted emoji similarity with the initial one or the one that was in the users head. So, we might need to carry out an extrinsic evaluation in the end to find out its actual performance.

