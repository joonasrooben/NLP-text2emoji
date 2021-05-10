# caption2emoji
**Aima Allik, Joonas Järve, Shumpei Morimoto**

### Objective
The plan is to create a NLP model that is capable to assign/predict emojis to given caption/tweet/text. 
To illustrate the idea, let's think of a typical predictive keyboard that suggest us the next words. This is very similar to our goal, the model will predict an emoji corresponding to the inserted text. The idea is partly based on the article [[1]](#1) by Francesco Barbieri et al.

### Evaluation
In most of the articles, goodness is usually measured with traditional recall, precision, accuracy and F-score. In addition to the latter measures, as the target is rather subjective we cannot really tell which is the gold label or if we are not assigning too big of an error to the wrong label, although some emojis are very similar (e.g :smiley: and :smile:). Therefore, we could use a different measure. One way is to measure the predicted emoji similarity with the initial description of the emoji. The other way would be to evaluate the prediction which was in the users head. So, for that, we might need to carry out an extrinsic evaluation in the end to find out the model's actual performance.

### Related work
In this section we will give a slight overview of what has been done earlier and mention the papers' similarities to our project. Some of the papers are more related to our project but some of the chosen ones face the wide world of emoji research from the different perspective i.e embeddings perspective. Nevertheless, we included them, as they point out the possibilities of the area.

* [Are Emojis Predictable?](https://arxiv.org/pdf/1702.07285.pdf)[[1]](#1)
After some additional research, we found a paper that very much does exactly what we are planning to do. In a way it was a good finding as it will provide us some more useful insights to the problem. The authors train a bidirectional LSTM to predict emojis based on tweets. They use word and character-based embeddings to represent tokens. The dataset consists of half a million tweets with only 20 different emojis. As a baseline model, they use *Bag of Words* and *skip-gram vector average* models. The best results were achieved with character-based embeddings and b-LSTM. In contrary to this article, we plan to try to concatenate the word- and character-based embeddings, use GRU instead and try with CNN as well. The amount to emojis taken to the project is still an open question from our side.

* [Multimodal Emoji Prediction](https://arxiv.org/pdf/1803.02392.pdf)[[2]](#2)
This paper is an advancement of the previous *Are Emojis Predictable?* by not only taking account the text but also the picture. For such task, authors used Instagram posts. They used a smaller dataset but also with pictures. They used the previous article model as a baseline and ResNet based model alongside with the *FastText* model. They showed that additional visual information provides slightly better results. In our work, we will stick to text and not include any visual extras as the improvement were not so vast.

* [Emojitalianobot and EmojiWorldBot](http://ceur-ws.org/Vol-1749/paper37.pdf)[[3]](#3)
The authors of this paper have created a bot that translates textual language to emojis and vice versa. This is rather different from our goal but has brought attention to extrinsic evaluation means such as *crowdscourcing*. They also mention the annotations that provide labels for Unicode characters i.e [emojis](https://www.unicode.org/cldr/cldr-aux/charts/29/annotations/uralic.html) in our case.

* [A Semantics-Based Measure of Emoji Similarity](https://arxiv.org/pdf/1707.04653.pdf)[[4]](#4)
This paper analyzed the semantic similarity of emoji through embedding models using the word embedding model, which is called the emoji embedding model. Emojis were extracted into machine-readable emoji description(called sense) from EmojiNet. The author of this paper created a new dataset called *EmoSim508* to evaluate the emoji embedding models' performance. EmoSim508 is a human-annotated semantic similarity score of 508 carefully selected emoji pairs. The emoji embedding model learned on sense labels, correlate best with the created dataset's emoji similarity ranking. The paper also shows the result of the real-world use-case of the emoji embedding models, which is used in a sentiment analysis task.

* [Emoji Representations from their Description](https://arxiv.org/pdf/1609.08359.pdf)[[5]](#5)
In this paper, the representation of emojis were estimated directly and only from their Unicode description, whereas we plan to use tweets as the main training data. The plus side of using tweets is that the model will also be prepared for not so perfect scenarios like the usage of sarcasm. As in some previously mentioned papers, they also used the *Bag of Words* approach for encoding descriptions. They released *emoji2vec* which has embeddings of 1661 emoji symbols, but in our project, we will lack the chance of using them as the emoji itself will not be included in the training set.  

* [Emoji Prediction: Extensions and Benchmarking](https://arxiv.org/ftp/arxiv/papers/2007/2007.07389.pdf)[[6]](#6)
The goal of the paper is very similar to ours - to predict the most appropriate emoji(s) to a given piece of text. They used the BERT and DeepMoji models for prediction. The authors address the problem of the lack of a benchmark dataset in the field of emoji prediction. They state that the insufficent means make it difficult to compare the performances of different emoji prediction models. Therefore, they provide such benchmark datasets and are waiting for other researchers to base their work on the datasets. As this article is fairly new (from August of 2020), it is a good opportunity for us to make use of the data. The only downside seems to be that the data and code for this paper is available only upon request.

### Data

The main dataset has 1M sentences with emojis from Twitter from about 2017-Jan (obtained from https://github.com/jiali-ms/emoji2vec). The second dataset is MC-20 (obtained from the authors of “Emoji Prediction: Extensions and Benchmarking”), this dataset consists of the 20 most frequently used emojis with according tweets. These emojis are ❤, 😂, 👍, 🙏, 🙌, 😘, 😍, 😊, 🔥, 👏, 👌, 💪, 👊, 😉, 🎉, 😎, 😁, 💯, 😜 and 👀. In the first dataset, the distribution of emojis is by the actual frequency whereas in the second dataset, the distribution of each emoji is equal. 

The first Twitter dataset is downsized to consist only the same 20 most frequent emojis and match the size of the second dataset (total of 200000 tweets). So in the first dataset there are almost 70000 tweets for the most frequent emoji and 1500 tweets for the least frequent and for the second dataset there are 10000 tweets for each of the 20 emojis. Both of the datasets are split into 80% train, 10% dev and 10% test. In case of computational limitations on training the models, we will downsize both datasets equally.


### Method

The initial idea of building a RNN or CNN has been revised due to the last practice session where we covered BERT model. As, paper [[1]](#1) can provide us a very good comparison, we will definately try to implement GRU based recurrent network. In addition, we will create a BOW baseline model TF-IDF and hopefully an advancement to the RNN model- BERT. 

For TF-IDF, BERT and RNN implementations- we will use the materials from the homeworks as they provide a very good starting point for our own specific implementation. So, we will not write our code from scratch and can more concentrate on adaption, performance and evaluation. Therefore we will use *PyTorch* framework.

**BiGRU** based model. We will use three types of embeddings: character based, word based pretrained and word based trainable embeddings. The use of character based embeddings is very much necessary as we are dealing with texts from social media which do not tend to be grammatically correct nor standard. The emedded sentences will be sent to bidirectional GRU layer and on top of it we will use two linear layers with Relu activation functions. Dropout might be also considered regularization. In the end we will have a softmax layer which outputs the probability of the correct emoji. Here is a figure of the structure as well.  
![*BiGRU* based model](gru.png)

**BERT** based model. We will train a multi-class classifier by fine-tuning BERT from transformers library by HuggingFace. We will not use the large BERT but the light-weight version *DistilBert* with AdamW optimizer. 

As related work does not give a plain signal if balanced or unbalanced data is better in emoji prediction, all the models will be trained on two datasets: unbalanced and balanced with same size and emojis. This way we hope to draw some conclusions in this dark area.

In the evaluation phase, we plan to output F1-score, recall, precision and accuracy. In addition, as emoji-prediciton can be rather fuzzy- we will try to take it in account by using also the Mean Reciprocal Rank (MRR) usually used in evaluating of the factoid QA models. But, as our task is as subjective- it should give a better understanding of the models' goodness. 

### Preliminary results

#### BERT
Two BERT models were trained- one with the Twitter unbalanced dataset and other with MC-20 balanced dataset. During fine tuning, learning rate 5e-5 and batch size 16 with 3 epochs of training gave the best results. 

| Results  | Tf_idf:Twitter unbalanced | Tf_idf:MC_20 | BERT:Twitter unbalanced | BERT:MC_20 |
|--------:|--------------------|------:|------:|------:|
|Accuracy | 0.276 | 0.876 | 0.88 | 0.46|
|Recall   | 0.168 |   0.875  | 0.880 | 0.12 |
|Precision|  0.162 | 0.935 | 0.939 | 0.20 |
| F_score | 0.164 | 0.894 | 0.900 | 0.14 |
|  MRR    |  0.413 | 0.891 | 0.908 |  0.601 |

|    |       | MC_20     | MC_20    | MC_20    | Twitter   | Twitter   | Twitter   |
|----|-------|-----------|----------|----------|-----------|-----------|-----------|
|    | Emoji | precision |   recall |  f_score | precision |    recall |   f_score |
|  0 | 💪     |  0.985765 | 0.848825 | 0.912184 |  0.454545 |  0.246914 |      0.32 |
|  1 | 👊     |  0.967337 | 0.805439 | 0.878995 |  0.371134 |  0.169014 |  0.232258 |
|  2 | 🙌     |  0.987624 |     0.84 |  0.90785 |  0.407407 | 0.0531401 | 0.0940171 |
|  3 | 🙏     |  0.985366 | 0.805583 | 0.886451 |  0.491736 |  0.445693 |  0.467583 |
|  4 | ❤     |  0.961085 | 0.842813 | 0.898072 |  0.626168 |  0.241877 |  0.348958 |
|  5 | 😜     |  0.355384 | 0.955535 | 0.518082 |         0 |         0 |         0 |
|  6 | 👏     |   0.98029 |  0.89404 | 0.935181 |  0.457143 |  0.268156 |  0.338028 |
|  7 | 🔥     |  0.852284 | 0.886754 | 0.869177 |  0.651515 |  0.552463 |  0.597914 |
|  8 | 😎     |  0.995386 | 0.893375 | 0.941626 |  0.417021 |  0.168966 |  0.240491 |
|  9 | 😁     |  0.986111 |   0.8841 | 0.932323 |  0.368421 | 0.0127042 | 0.0245614 |
| 10 | 😉     |  0.965632 | 0.897013 | 0.930059 |  0.322129 |   0.15928 |   0.21316 |
| 11 | 🎉     |  0.987395 | 0.914397 | 0.949495 |  0.542955 |  0.714932 |  0.617187 |
| 12 | 💯     |  0.989035 | 0.873185 | 0.927506 |  0.540625 |  0.480556 |  0.508824 |
| 13 | 👌     |  0.994094 | 0.927456 |  0.95962 |  0.422794 |  0.282209 |  0.338484 |
| 14 | 😘     |  0.956212 | 0.954268 | 0.955239 |   0.46496 |  0.389391 |  0.423833 |
| 15 | 👍     |  0.994813 | 0.895425 | 0.942506 |  0.453654 |  0.487548 |  0.469991 |
| 16 | 👀     |   0.98893 | 0.885463 |  0.93434 |  0.548558 |  0.488184 |  0.516613 |
| 17 | 😊     |  0.980794 | 0.836245 |  0.90277 |  0.376783 |  0.456224 |  0.412716 |
| 18 | 😍     |  0.996629 | 0.933684 |  0.96413 |  0.542152 |   0.63398 |  0.584482 |
| 19 | 😂     |  0.877944 | 0.835882 | 0.856397 |  0.663419 |  0.853464 |  0.746537 |

**MC_20 model**: 

|    | Emoji   |   precision |   recall |   f_score |
|---:|:--------|------------:|---------:|----------:|
|  0 | 💪      |    0.985765 | 0.848825 |  0.912184 |
|  1 | 👊      |    0.967337 | 0.805439 |  0.878995 |
|  2 | 🙌      |    0.987624 | 0.84     |  0.90785  |
|  3 | 🙏      |    0.985366 | 0.805583 |  0.886451 |
|  4 | ❤       |    0.961085 | 0.842813 |  0.898072 |
|  5 | 😜      |    0.355384 | 0.955535 |  0.518082 |
|  6 | 👏      |    0.98029  | 0.89404  |  0.935181 |
|  7 | 🔥      |    0.852284 | 0.886754 |  0.869177 |
|  8 | 😎      |    0.995386 | 0.893375 |  0.941626 |
|  9 | 😁      |    0.986111 | 0.8841   |  0.932323 |
| 10 | 😉      |    0.965632 | 0.897013 |  0.930059 |
| 11 | 🎉      |    0.987395 | 0.914397 |  0.949495 |
| 12 | 💯      |    0.989035 | 0.873185 |  0.927506 |
| 13 | 👌      |    0.994094 | 0.927456 |  0.95962  |
| 14 | 😘      |    0.956212 | 0.954268 |  0.955239 |
| 15 | 👍      |    0.994813 | 0.895425 |  0.942506 |
| 16 | 👀      |    0.98893  | 0.885463 |  0.93434  |
| 17 | 😊      |    0.980794 | 0.836245 |  0.90277  |
| 18 | 😍      |    0.996629 | 0.933684 |  0.96413  |
| 19 | 😂      |    0.877944 | 0.835882 |  0.856397 |

test with Twitter data:
|    | Emoji   |   precision |    recall |    f_score |
|---:|:--------|------------:|----------:|-----------:|
|  0 | 💪      |  0.00625    | 0.034965  | 0.0106045  |
|  1 | 👊      |  0.00480962 | 0.0594059 | 0.00889878 |
|  2 | 🙌      |  0.0186514  | 0.0546218 | 0.0278075  |
|  3 | 🙏      |  0.089404   | 0.1875    | 0.121076   |
|  4 | ❤       |  0.025      | 0.0833333 | 0.0384615  |
|  5 | 😜      |  0.027027   | 0.0712074 | 0.0391823  |
|  6 | 👏      |  0.0136635  | 0.0471976 | 0.0211921  |
|  7 | 🔥      |  0.0105769  | 0.0219124 | 0.0142672  |
|  8 | 😎      |  0.0443828  | 0.0597015 | 0.0509149  |
|  9 | 😁      |  0.0164671  | 0.0206379 | 0.0183181  |
| 10 | 😉      |  0.0342707  | 0.0562771 | 0.0425997  |
| 11 | 🎉      |  0.0210325  | 0.0154278 | 0.0177994  |
| 12 | 💯      |  0.0366379  | 0.0226365 | 0.0279835  |
| 13 | 👌      |  0.0370611  | 0.0694275 | 0.0483256  |
| 14 | 😘      |  0.0583717  | 0.044186  | 0.0502978  |
| 15 | 👍      |  0.0589722  | 0.0661626 | 0.0623608  |
| 16 | 👀      |  0.0666014  | 0.0453333 | 0.0539468  |
| 17 | 😊      |  0.038055   | 0.021805  | 0.0277243  |
| 18 | 😍      |  0.0703518  | 0.0600214 | 0.0647773  |
| 19 | 😂      |  0.395156   | 0.0795375 | 0.132421   |

 

**Twitter Model**

 |    | Emoji   |   precision |    recall |   f_score |
|---:|:--------|------------:|----------:|----------:|
|  0 | 💪      |    0.454545 | 0.246914  | 0.32      |
|  1 | 👊      |    0.371134 | 0.169014  | 0.232258  |
|  2 | 🙌      |    0.407407 | 0.0531401 | 0.0940171 |
|  3 | 🙏      |    0.491736 | 0.445693  | 0.467583  |
|  4 | ❤       |    0.626168 | 0.241877  | 0.348958  |
|  5 | 😜      |    0        | 0         | 0         |
|  6 | 👏      |    0.457143 | 0.268156  | 0.338028  |
|  7 | 🔥      |    0.651515 | 0.552463  | 0.597914  |
|  8 | 😎      |    0.417021 | 0.168966  | 0.240491  |
|  9 | 😁      |    0.368421 | 0.0127042 | 0.0245614 |
| 10 | 😉      |    0.322129 | 0.15928   | 0.21316   |
| 11 | 🎉      |    0.542955 | 0.714932  | 0.617187  |
| 12 | 💯      |    0.540625 | 0.480556  | 0.508824  |
| 13 | 👌      |    0.422794 | 0.282209  | 0.338484  |
| 14 | 😘      |    0.46496  | 0.389391  | 0.423833  |
| 15 | 👍      |    0.453654 | 0.487548  | 0.469991  |
| 16 | 👀      |    0.548558 | 0.488184  | 0.516613  |
| 17 | 😊      |    0.376783 | 0.456224  | 0.412716  |
| 18 | 😍      |    0.542152 | 0.63398   | 0.584482  |
| 19 | 😂      |    0.663419 | 0.853464  | 0.746537  |

mc_20 data:
acc:0.0496866382552018
mrrr: 0.1779
0.044067249605117556 0.050490362936857644 0.034572703873399516
|    | Emoji   |   precision |     recall |    f_score |
|---:|:--------|------------:|-----------:|-----------:|
|  0 | 💪      |  0.00507614 | 0.00095057 | 0.00160128 |
|  1 | 👊      |  0.0108696  | 0.00298211 | 0.00468019 |
|  2 | 🙌      |  0.0172414  | 0.00106496 | 0.00200602 |
|  3 | 🙏      |  0.311321   | 0.106223   | 0.1584     |
|  4 | ❤       |  0.0432432  | 0.00815494 | 0.0137221  |
|  5 | 😜      |  0          | 0          | 0          |
|  6 | 👏      |  0.0267983  | 0.0182517  | 0.0217143  |
|  7 | 🔥      |  0.0141844  | 0.0055814  | 0.00801068 |
|  8 | 😎      |  0.0434783  | 0.0125918  | 0.0195281  |
|  9 | 😁      |  0          | 0          | 0          |
| 10 | 😉      |  0.019802   | 0.00392157 | 0.00654664 |
| 11 | 🎉      |  0.0304102  | 0.0434783  | 0.0357886  |
| 12 | 💯      |  0.0422961  | 0.0140562  | 0.0211002  |
| 13 | 👌      |  0.036105   | 0.0321951  | 0.0340382  |
| 14 | 😘      |  0.0558767  | 0.0290873  | 0.0382586  |
| 15 | 👍      |  0.0518559  | 0.0909091  | 0.066041   |
| 16 | 👀      |  0.036791   | 0.0757098  | 0.0495186  |
| 17 | 😊      |  0.0249878  | 0.0521472  | 0.033786   |
| 18 | 😍      |  0.0448573  | 0.105207   | 0.0628971  |
| 19 | 😂      |  0.0661511  | 0.407295   | 0.113817   |



#### TF-IDF

| TF-IDF  | Twitter unbalanced | MC_20 |
|--------:|--------------------|------:|
|Accuracy | 0.276 | 0.876 |
|Recall   | 0.168 |   0.875  |
|Precision|  0.162 | 0.935 |
| F_score | 0.164 | 0.894 |
|  MRR    |  0.413 | 0.891 |


|    | Emoji | precision_mc | recall_mc | f_score_mc | precision_tw | recall_tw | f_score_tw |
|---:|-------|-------------:|----------:|-----------:|-------------:|----------:|-----------:|
|  0 | ❤     |        0.984 |     0.832 |      0.901 |        0.105 |     0.122 |      0.113 |
|  1 | 😂     |        0.967 |     0.799 |      0.875 |         0.52 |     0.446 |       0.48 |
|  2 | 👍     |        0.978 |     0.845 |      0.907 |        0.173 |     0.165 |      0.169 |
|  3 | 🙏     |        0.964 |     0.815 |      0.883 |        0.185 |     0.173 |      0.179 |
|  4 | 🙌     |        0.867 |     0.841 |      0.854 |        0.089 |       0.1 |      0.094 |
|  5 | 😘     |        0.977 |     0.765 |      0.858 |        0.172 |     0.191 |      0.181 |
|  6 | 😍     |        0.977 |     0.886 |      0.929 |        0.248 |     0.304 |      0.273 |
|  7 | 😊     |        0.949 |      0.86 |      0.902 |        0.171 |     0.191 |      0.181 |
|  8 | 🔥     |        0.996 |     0.914 |      0.953 |        0.241 |      0.27 |      0.255 |
|  9 | 👏     |        0.957 |     0.895 |      0.925 |        0.101 |     0.099 |        0.1 |
| 10 | 👌     |        0.982 |     0.908 |      0.944 |        0.126 |     0.142 |      0.133 |
| 11 | 💪     |        0.996 |      0.91 |      0.951 |        0.044 |     0.085 |      0.058 |
| 12 | 👊     |        0.969 |     0.863 |      0.913 |          0.1 |     0.117 |      0.108 |
| 13 | 😉     |        0.996 |     0.923 |      0.958 |         0.09 |     0.086 |      0.088 |
| 14 | 🎉     |        0.981 |      0.93 |      0.955 |        0.347 |     0.359 |      0.353 |
| 15 | 😎     |        0.986 |     0.892 |      0.937 |        0.077 |     0.069 |      0.073 |
| 16 | 😁     |        0.982 |     0.886 |      0.932 |        0.054 |     0.049 |      0.051 |
| 17 | 💯     |        0.336 |     0.974 |        0.5 |        0.164 |     0.146 |      0.154 |
| 18 | 😜     |        0.988 |     0.952 |       0.97 |        0.047 |     0.038 |      0.042 |
| 19 | 👀     |        0.864 |     0.815 |      0.839 |         0.18 |     0.204 |      0.191 |


**Evaluation by emoji definitions**

|    | Gold   | Tf-idf_twitter |Tf-idf_MC_20|BERT_MC_20 | BERT_twitter | Line                                                                                        |
|---:|:-------|:---------|:-----------|:--------------------------------------------------------------------------------------------|
|  0 | 🙏     | 👍   | 👀    | 🙏 | 😊 |Two hands placed firmly together, meaning please or thank you in Japanese culture           |
|  1 | 🙌     | 😂   | 👍   | 😘 |🎉 |Two hands raised in the air, celebrating success or another joyous event                    |
|  2 | 😜     | 😉   | 🎉  |😍 | 😂 | A face showing a stuck-out tongue, winking at the same time                                 |
|  3 | 😘     | 😘   | 😘  |😎 | 😂 | An emoji face blowing a kiss; but officially called “Face Throwing A Kiss”                  |
|  4 | 😎     | 😍   | 👀  | 👍 | 😎 |A face smiling and wearing dark sunglasses that is used to denote a sense of cool           |
|  5 | 😍     | 👀   | 👀 | 😉| 😍 |A face with hearts instead of eyes, or Heart Eyes Emoji as it is generally known            |
|  6 | 😊     | ❤    | 😁  |🔥| 😊 | A smiling face, with smiling eyes and rosy cheeks                                           |
|  7 | 😉     | 😉   | 👌  |🔥| 😊 |A classic winky emoji; winking and smiling                                                  |
|  8 | 😂     | 👌    | 💯  |😍| 😂 |A laughing emoji which at small sizes is often mistaken for being tears of sadness          |
|  9 | 😁     | 😍   | 😁  | 😂 | 😂 |A version of the grinning face showing smiling eyes                                         |
| 10 | ❤      | 😍   | 👀  | 👏 | 👏 |A classic red love heart emoji, used to express love                                        |
| 11 | 🔥     | 😍    | 👍  | 😎 | 🔥 |A small flame, mostly yellow but red at the top                                             |
| 12 | 💯     | 😂   | 💯  | ❤ | 👌 | 100 emoji: the number one-hundred, written in red, underlined twice for emphasis            |
| 13 | 💪     | 👀   | 🔥  | 😂|  💪 |An arm flexing to show its biceps muscle                                                    |
| 14 | 👏     | 👀   | 👀 | 😁 |👏 | Two hands clapping emoji, which when used multiple times can be used as a round of applause |
| 15 | 👍     | 👍   | 👏  | 🙌 | 😊 |A thumbs-up gesture indicating approval                                                     |
| 16 | 👌     | 😂   | 👏  | ❤ | 😂 |Index finger touching thumb to make an open circle                                          |
| 17 | 👊     | 👊   | 👊  | 😂 | 👊 |A fist displayed in a position to punch someone, or to fist-bump another person             |
| 18 | 👀     | 😍   |👀  | 😂 | 👀 |A pair of eyes, glancing slightly to the left on most platforms                             |
| 19 | 🎉     | 🎉   | 😍 | 👍 | 🎉 |A colorful party popper, used for party or other celebration                                |


## References
<a id="1">[1]</a> 
Barbieri, Francesco & Ballesteros, Miguel & Saggion, Horacio. (2017). Are Emojis Predictable?. 105-111. 10.18653/v1/E17-2017. 

<a id="2">[2]</a> 
Barbieri, Francesco & Ballesteros, Miguel & Ronzano, Francesco & Saggion, Horacio. (2018). Multimodal Emoji Prediction. 679-686. 10.18653/v1/N18-2107. 

<a id="3">[3]</a> 
Monti, Johanna & Sangati, Federico & Chiusaroli, Francesca & Benjamin, Martin & Mansour, Sina. (2016). Emojitalianobot and EmojiWorldBot. 10.4000/books.aaccademia.1811. 

<a id="4">[4]</a> 
Wijeratne, Sanjaya & Balasuriya, Lakshika & Sheth, Amit & Doran, Derek. (2017). A Semantics-Based Measure of Emoji Similarity. 10.1145/3106426.3106490. 

<a id="5">[5]</a> 
Eisner, Ben & Rocktäschel, Tim & Augenstein, Isabelle & Bosnjak, Matko & Riedel, Sebastian. (2016). emoji2vec: Learning Emoji Representations from their Description. 48-54. 10.18653/v1/W16-6208. 

<a id="6">[6]</a> 
Ma, Weicheng & Liu, Ruibo & Wang, Lili & Vosoughi, Soroush. (2020). Emoji Prediction: Extensions and Benchmarking. 
