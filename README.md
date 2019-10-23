# Multiple Relational Facts Extraction

## Background
### What is relational fact?

A relational fact includes a head entity, a tail entity and an relation. A relational fact is also called triplet. For example, <New York, located_in, America> is a relational fact and "New York" is the head entity and "America" is the tail entity. "located_in" is the relation between these two entities. The head entity is also called the first entity (e1) while the tail entity called the second entity (e2). Mostly, the entity pair (e1, e2) is different from (e2, e1).

### What is relation extraction?

There are different definitions of relation extraction. However, the core concept is the same: extracting triplets from text. Different definitions may contains different constraints. Most relation extraction works assume that the text as a natural language sentence, the entity is a word or phrase in this sentence and the relations are predefined. The major difference between relation classification and relation extraction is if the entity pair are given. If the entity pair is given, the relation extraction task is turned into a relation classification task. For the relation extraction task, there could be multiple entities in a sentence, which leads to multiple relational facts. Therefore, we call this task as multiple relational fats extraction. 
Most relation extraction works assume that one entity may participate in only one triplet, however, they neglect the so called "overlapping problem". What's more, since there are multiple relational facts to be extracted, which order should we follow to extract them? This problem is called the "extraction order problem".

### Handle the overlapping problem

To handle the overlapping problem, we propose to use sequence to sequence model with copy mechanism to generate triplets directly.
This work has been published in ACL2018: [Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism](https://github.com/xiangrongzeng/multi_re/blob/master/paper/Extracting-relational-facts-by-an-end-to-end-neural-model-with-copy-mechanism.pdf).
We denote this work as copy_re.

<img src="https://github.com/xiangrongzeng/multi_re/blob/master/img/copy_re.jpg" width="400" alt="copy_re"/>


### Handle the extraction order problem

To handle the extraction order problem (and hanle the overlapping problem at the same time), we apply reinforcement learning on the above sequence to sequence model to learn a good extraction order automatically.
This work has been accepted by EMNLP2019: [Learning the Extraction Order of Multiple Relational Facts in a Sentence with Reinforcement Learning](https://github.com/xiangrongzeng/multi_re/blob/master/paper/Learning-the-Extraction-Order-of-Multiple-Relational-Facts-in-a-Sentence-with-Reinforcement-Learning.pdf).
We denote this work as order_copy_re.

<img src="https://github.com/xiangrongzeng/multi_re/blob/master/img/order_copy_re.jpg" width="400" alt="order_copy_re"/>

## Code and Data

The code is released for both papers. The code and data for the overlapping problem was released in [copy_re](https://github.com/xiangrongzeng/copy_re). We rewrite the copy_re code so that RL can be added in. The order_copy_re code is mostly the same with copy_re, we use different loss (RL loss to be specific). The released code can be used for either copy_re or order_copy_re with different settings.

### Data

The data (including the data preprocessing) for both copy_re and order_copy_re are exactly the same.
You need to modify the data path in const.py before running the code.
The pre-processed data is released.

WebNLG:

 - [dataset](https://drive.google.com/open?id=1zISxYa-8ROe2Zv8iRc82jY9QsQrfY1Vj)
 - [pre-trained word embedding](https://drive.google.com/open?id=1LOT2-JxjjglCFyxv-JQAJlJvEmleSXZl)

NYT:

 - [dataset](https://drive.google.com/open?id=10f24s9gM7NdyO3z5OqQxJgYud4NnCJg3)
 - [pre-trained word embedding](https://drive.google.com/open?id=1yVjN-0lZid6YJmsX5g8x_YKiCfnRy8IL)
 
 Or, you can download them (without pre-trained word embedding) in [Baidu SkyDrive](https://pan.baidu.com/s/1BcbFmCvHGNfaiQDyDma-JA) with code "y286".

### Environment

- python2.7
- [requirements.txt](https://github.com/xiangrongzeng/copy_re/blob/master/requirements.txt)

### NLL train, valid and test

NLL Train means we train the model with negative log likelyhood loss. Copy_re uses NLL loss.
You can train the model with NLL loss as follows, the meaning of parameters can be found in ***argument_parser.py***

```
python main.py -a train -d nyt -l nll -m onedecoder -b 100 -tn 5 -lr 0.001 -en 20 -sf 2 -hn 1000 -n tmp -g 0 -cell gru -sobm 0
```

To valid or test the models, change the value of parameter **-a** to *valid* or *test*. 

The code will make a folder with the name contains most of parameter values. The model in different epochs will be stored in this folder. For example, the above settings will lead to a folder *nyt-ONE-NLL-5-0.001-100-FixedUnSorted-gru-1000-1000*. In this way, we can try many different settings without inference each other.

The parameter **-n** means the name of experiment. We create a folder with this name. All settings with the same experiment name will be placed in the same folder. Therefore, you can do exploratory experiments in **-n tmp** and run your best settings in **-n mybest**.

In a word, the directory will look like this:
```
--exp_name
----settings_name
------epoch_i.model
------epoch_j.model
......
----another_settings_name
------epoch_i.model
------epoch_j.model
......
--another_exp_name
......
```

### RL train, valid and test

Before we apply RL training, we need to pre-train the model with NLL loss first. We need to set the path of the pre-trained model parameters to initialize the model when apply RL training.

```
python main.py -a train -d nyt -l rl -m onedecoder -b 100 -tn 5 -lr 0.0005 -en 50 -sf 5 -hn 1000 -n tmp -g 0 -cell gru -re 5 -rip 'nyt-ONE-NLL-5-0.001-100-FixedUnSorted-gru-1000-1000'
```

To valid or test the models, change the value of parameter **-a** to *valid* or *test*. The parameter **-rip** is no longer necessary during valid or test.

Remember that the value of **-d, -m, -hn, -n, -cell**  must be the same with the initialized model.
 
