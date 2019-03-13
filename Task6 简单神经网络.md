
**1. 文本表示：从one-hot到word2vec。**

1.1 词袋模型：离散、高维、稀疏。

1.2 分布式表示：连续、低维、稠密。word2vec词向量原理并实践，用来表示文本。

**2. 走进FastText**

2.1 FastText的原理。 

2.2 利用FastText模型进行文本分类。

### Word2Vec

Word2Vec是google提出的一个学习word vecor(也叫word embedding)的框架。

它主要提出了两个模型结构CBOW和Skip-gram，这两个模型都属于Log Linear模型，结构如下所示：
![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

**CBOW对小型数据比较合适，而Skip-gram在大型语料中表现得更好。**

### CBOW模型

CBOW main idea：Predict center word from (bag of) context words.

cbow全称为continues bag of words，之所以称为continues是因为学习出来的word vector是连续的分布式表示的，而又称它是一个bag of words模型是因为在训练时，历史信息的词语顺序不影响Projection，因为之后不是将各个词concate，而是sum，而且这样也大大较少了计算量！

CBOW的详细结构如下：
![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)


**输入是one-hot向量表示的词，之后Projection成N维的低维稠密向量并加权求和，然后与W’相乘，经过softmax求得预测单词的概率分布y，概率分布中概率最大的单词就是应该输出的单词。**

### Skip-grams模型
![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

### 加速方法

为了提高word vector的质量，加快训练速度，Google提出了层次softmax和负采样的加速方法。普遍认为层次Softmax对低频词效果较好；负采样对高频词效果较好，向量维度较低时效果更好。

**1，层次softmax**

本质：把 N 分类问题变成 log(N)次二分类。

为了避免要计算所有词的softmax概率，word2vec采样了霍夫曼树来代替从隐藏层到输出softmax层的映射。

该方法不用为了获得最后的概率分布而评估神经网络中的W个输出结点，而只需要评估大约log2(W)个结点。层次Softmax使用一种二叉树结构来表示词典里的所有词，V个词都是二叉树的叶子结点，而这棵树一共有V−1个非叶子结点。一般采用二叉哈弗曼树，因为它会给频率高的词一个更短的编码。


**2，负采样**

本质：预测总体类别的一个子集。

让我们回顾一下训练的详细过程：每次用一个训练样本，就要更新所有的权重，这显然对于有大量参数和大量样本的模型来说，十分耗费计算量。但其实每次用一个训练样本时，我们并不需要更新全部的参数，我们只需要更新那部分与这个样本相关的参数即可。而负采样的思想就是每次用一个训练样本更新一小部分参数。负采样的意思是每次用的那个训练样本最后输出的值其实只有一个词的值是1，而其他不正确的词都是0，我们可以从那么是0的单词中采样一部分如5~20词来进行参数更新，而不使用全部的词。


## 利用gensim库实现word2vec

在gensim中，word2vec 相关的参数都在包gensim.models.word2vec中。完整函数如下：

             gensim.models.word2vec.Word2Vec(sentences=None,size=100,alpha=0.025,window=5, min_count=5, max_vocab_size=None, sample=0.001,seed=1, workers=3,min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>,iter=5,null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000）

　　　　1) sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。对于大语料集，建议使用BrownCorpus,Text8Corpus或lineSentence构建。

　　　　2) size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，视语料库的大小而定。

        3) alpha： 是初始的学习速率，在训练过程中会线性地递减到min_alpha。

　　　　4) window：即词向量上下文最大距离，skip-gram和cbow算法是基于滑动窗口来做预测。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。对于一般的语料这个值推荐在[5,10]之间。

　      5) min_count:：可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。

        6) max_vocab_size: 设置词向量构建期间的RAM限制，设置成None则没有限制。

        7) sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。

        8) seed：用于随机数发生器。与初始化词向量有关。

        9) workers：用于控制训练的并行数。

       10) min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每    轮的迭代步长可以由iter，alpha， min_alpha一起得出。对于大语料，需要对alpha, min_alpha,iter一起调参，来选                        择合适的三个值。

　　　　11) sg: 即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。

        12)hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。

　　　　13) negative:如果大于零，则会采用negativesampling，用于设置多少个noise words（一般是5-20）。

　　　　14) cbow_mean: 仅用于CBOW在做投影的时候，为0，则采用上下文的词向量之和，为1则为上下文的词向量的平均值。默认值也是1,不推荐修改默认值。

        15) hashfxn： hash函数来初始化权重，默认使用python的hash函数。

　　　　16) iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。

        17) trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）。

        18) sorted_vocab： 如果为1（默认），则在分配word index 的时候会先对单词基于频率降序排序。

        19) batch_words：每一批的传递给线程的单词的数量，默认为10000。



```python
#分词
import jieba.analyse
import jieba
import os

 
raw_data_path = 'F:/sougou_data/'
cut_data_path = 'F:/sougou_cutdata/'
stop_word_path = 'F:/sougou_cutdata/stopwords.txt'
 
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'rb').readlines()]
    return stopwords
 
def cut_word(raw_data_path, cut_data_path ):
    data_file_list = os.listdir(raw_data_path)
    corpus = ''
    temp = 0
    for file in data_file_list:
        with open(raw_data_path + file,'rb') as f:
            print(temp+1)
            temp +=1
            document = f.read()
            document_cut = jieba.cut(document, cut_all=False)
            # print('/'.join(document_cut))
            result = ' '.join(document_cut)
            corpus += result
          #  print(result)
    with open(cut_data_path + 'corpus.txt', 'w+', encoding='utf-8') as f:
        f.write(corpus)  # 读取的方式和写入的方式要一致
 
    stopwords = stopwordslist(stop_word_path)  # 这里加载停用词的路径
    with open(cut_data_path + 'corpus.txt', 'r', encoding='utf-8') as f:
        document_cut = f.read()
        outstr = ''
        for word in document_cut:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
 
    with open(cut_data_path + 'corpus1.txt', 'w+', encoding='utf-8') as f:
            f.write(outstr)  # 读取的方式和写入的方式要一致
 
if __name__ == "__main__":
    cut_word(raw_data_path, cut_data_path )
 
#word2vec
# -*- coding: utf-8 -*-
 
 
from gensim.models import word2vec
import logging
 
##训练word2vec模型
 
# 获取日志信息
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
 
# 加载分词后的文本，使用的是Text8Corpus类
 
sentences = word2vec.Text8Corpus(r'F:\sougou_cutdata\corpus.txt')
 
# 训练模型，部分参数如下
model = word2vec.Word2Vec(sentences, size=100, hs=1, min_count=1, window=3)
 
# 模型的预测
print('-----------------分割线----------------------------')
 
# 计算两个词向量的相似度
try:
    sim1 = model.similarity(u'中央企业', u'事业单位')
    sim2 = model.similarity(u'教育网', u'新闻网')
except KeyError:
    sim1 = 0
    sim2 = 0
print(u'中央企业 和 事业单位 的相似度为 ', sim1)
print(u'人民教育网 和 新闻网 的相似度为 ', sim2)
 
print('-----------------分割线---------------------------')
# 与某个词（李达康）最相近的3个字的词
print(u'与国资委最相近的3个字的词')
req_count = 5
for key in model.similar_by_word(u'国资委', topn=100):
    if len(key[0]) == 3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break
 
print('-----------------分割线---------------------------')
# 计算某个词(侯亮平)的相关列表
try:
    sim3 = model.most_similar(u'新华社', topn=20)
    print(u'和 新华社 与相关的词有：\n')
    for key in sim3:
        print(key[0], key[1])
except:
    print(' error')
 
print('-----------------分割线---------------------------')
# 找出不同类的词
sim4 = model.doesnt_match(u'新华社 人民教育出版社 人民邮电出版社 国务院'.split())
print(u'新华社 人民教育出版社 人民邮电出版社 国务院')
print(u'上述中不同类的名词', sim4)
 
print('-----------------分割线---------------------------')
# 保留模型，方便重用
model.save(u'搜狗新闻.model')
 
# 对应的加载方式
# model2 = word2vec.Word2Vec.load('搜狗新闻.model')
# 以一种c语言可以解析的形式存储词向量
# model.save_word2vec_format(u"书评.model.bin", binary=True)
# 对应的加载方式
# model_3 =word2vec.Word2Vec.load_word2vec_format("text8.model.bin",binary=True)
 
 


```

## fasttext
fasttext是facebook开源的一个词向量与文本分类工具，在2016年开源，典型应用场景是“带监督的文本分类问题”。提供简单而高效的文本分类和表征学习的方法，性能比肩深度学习而且速度更快。

fastText结合了自然语言处理和机器学习中最成功的理念。这些包括了使用词袋以及n-gram袋表征语句，还有使用子字(subword)信息，并通过隐藏表征在类别间共享信息。我们另外采用了一个softmax层级(利用了类别不均衡分布的优势)来加速运算过程。

这些不同概念被用于两个不同任务：

    有效文本分类 ：有监督学习
    学习词向量表征：无监督学习

举例来说：fastText能够学会“男孩”、“女孩”、“男人”、“女人”指代的是特定的性别，并且能够将这些数值存在相关文档中。然后，当某个程序在提出一个用户请求（假设是“我女友现在在儿？”），它能够马上在fastText生成的文档中进行查找并且理解用户想要问的是有关女性的问题



fastText 方法包含三部分：模型架构、层次 Softmax 和 N-gram 特征

fastText 模型输入一个词的序列（一段文本或者一句话)，输出这个词序列属于不同类别的概率。 

序列中的词和词组组成特征向量，特征向量通过线性变换映射到中间层，中间层再映射到标签。

fastText 在预测标签时使用了非线性激活函数，但在中间层不使用非线性激活函数。 

fastText 模型架构和 Word2Vec 中的 CBOW 模型很类似。不同之处在于，fastText 预测标签，
而 CBOW 模型预测中间词

**FastText词向量优势**

（1）适合大型数据+高效的训练速度：能够训练模型“在使用标准多核CPU的情况下10分钟内处理超过10亿个词汇”，特别是与深度模型对比，fastText能将训练时间由数天缩短到几秒钟。使用一个标准多核 CPU，得到了在10分钟内训练完超过10亿词汇量模型的结果。此外， fastText还能在五分钟内将50万个句子分成超过30万个类别。

（2）支持多语言表达：利用其语言形态结构，fastText能够被设计用来支持包括英语、德语、西班牙语、法语以及捷克语等多种语言。它还使用了一种简单高效的纳入子字信息的方式，在用于像捷克语这样词态丰富的语言时，这种方式表现得非常好，这也证明了精心设计的字符 n-gram 特征是丰富词汇表征的重要来源。FastText的性能要比时下流行的word2vec工具明显好上不少，也比其他目前最先进的词态词汇表征要好。


```python
# _*_coding:utf-8 _*_
import logging
import time
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fasttext
#训练模型
start=time.clock()
classifier = fasttext.supervised("news_fasttext_train_2.txt","news_fasttext.model",label_prefix="__label__")
end=time.clock()
total_time=end-start
print("总耗时:"+str(total_time))
#load训练好的模型
#classifier = fasttext.load_model('news_fasttext.model.bin', label_prefix='__label__')
#测试模型
#load训练好的模型
#import fasttext
result = classifier.test("news_fasttext_test_2.txt")
print(result.precision)
print(result.recall)
```

利用fasttext对清华大学THUCNews中文语料进行分类

## fastText和word2vec的区别
**相似处：**

1.图模型结构很像，都是采用embedding向量的形式，得到word的隐向量表达。

2.都采用很多相似的优化方法，比如使用Hierarchical softmax优化训练和预测中的打分速度。

**不同处：**

1.模型的输出层：word2vec的输出层，对应的是每一个term，计算某term的概率最大；而fasttext的输出层对应的是 分类的label。不过不管输出层对应的是什么内容，起对应的vector都不会被保留和使用

2.模型的输入层：word2vec的输出层，是 context window 内的term；而fasttext对应的整个sentence的内容，包括term，也包括 n-gram的内容；

两者本质的不同，体现在 h-softmax的使用：

Wordvec的目的是得到词向量，该词向量最终是在输入层得到，输出层对应的 h-softmax也会生成一系列的向量，但最终都被抛弃，不会使用。
fasttext则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label（一个或者N个）


