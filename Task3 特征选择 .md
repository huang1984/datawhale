
# 特征选择

**Task3 特征选择  **

1. TF-IDF原理。

2. 文本矩阵化，使用词袋模型，以TF-IDF特征值为权重。（可以使用Python中TfidfTransformer库）

3. 互信息的原理。

4. 使用第二步生成的特征矩阵，利用互信息进行特征筛选。

5. 参考:

     文本挖掘预处理之TF-IDF：[文本挖掘预处理之TF-IDF - 刘建平Pinard - 博客园](https://www.cnblogs.com/pinard/p/6693230.html)

     使用不同的方法计算TF-IDF值：[使用不同的方法计算TF-IDF值 - 简书 ](https://www.jianshu.com/p/f3b92124cd2b)

     sklearn-点互信息和互信息：[sklearn：点互信息和互信息 - 专注计算机体系结构 - CSDN博客](https://blog.csdn.net/u013710265/article/details/72848755)
     如何进行特征选择（理论篇）机器学习你会遇到的“坑”：[如何进行特征选择（理论篇）机器学习你会遇到的“坑”](https://baijiahao.baidu.com/s?id=1604074325918456186&wfr=spider&for=pc)

## TF-IDF 原理

###  TF-IDF 介绍


 **TF-IDF(Term Frequency-Inverse Document Frequency, 词频-逆文件频率) 是一种用于资讯检索与资讯探勘的常用加权技术。TF-IDF是一种统计方法，用以评估 一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。**

###    TF-IDF 含义

 词频 (term frequency, TF)

 词频 (term frequency, TF) 指的是某一个给定的词语在该文件中出现的次数。这个数字通常会被归一化(一般是词频除以文章总词数), 以防止它偏向长的文件。（同一个词语在长文件里可能会比短文件有更高的词频，而不管该词语重要与否。）


    但是, 需要注意, 一些通用的词语对于主题并没有太大的作用, 反倒是一些出现频率较少的词才能够表达文章的主题, 所以单纯使用是TF不合适的。权重的设计必须满足：一个词预测主题的能力越强，权重越大，反之，权重越小。所有统计的文章中，一些词只是在其中很少几篇文章中出现，那么这样的词对文章的主题的作用很大，这些词的权重应该设计的较大。IDF就是在完成这样的工作

![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

### TF-IDF 实现

在scikit-learn中，有两种方法进行TF-IDF的预处理：

**第一种方法是在用CountVectorizer类向量化之后再调用TfidfTransformer类进行预处理。**

**第二种方法是直接用TfidfVectorizer完成向量化与TF-IDF预处理。**


首先我们来看第一种方法，CountVectorizer+TfidfTransformer的组合，代码如下：


```python
from sklearn.feature_extraction.text import TfidfVectorizer
 
corpus = ["TF-IDF是非常常用的文本挖掘预处理基本步骤，但是如果预处理中使用了Hash Trick，则一般就无法使用TF-IDF了，因为Hash Trick后我们已经无法得到哈希后的各特征的IDF的值。使用了IF-IDF并标准化以后，我们就可以使用各个文本的词特征向量作为文本的特征，进行分类或者聚类分析。当然TF-IDF不光可以用于文本挖掘，在信息检索等很多领域都有使用。因此值得好好的理解这个方法的思想。"]
 
# max_features是最大特征数
# min_df是词频低于此值则忽略，数据类型为int或float
# max_df是词频高于此值则忽略，数据类型为Int或float
tfidf_model = TfidfVectorizer(max_features=5, min_df=2, max_df=5).fit_transform(corpus)
print(tfidf_model.todense())

```

第一种https://mp.weixin.qq.com/s/7BG142jBb3K8kyJh_lf7gg


```python
      #encoding=utf-8
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
 
if __name__ == "__main__":
    docs_list=[]
    mywordlist=[]
    stopwords_path = "resource/stop_words.txt" # 停用词词表

    # 读取文件
    file_object = open('resource/text1.txt','r')
    try:
      for line in file_object:
          # 文本分词
          seg_list = jieba.cut(line, cut_all=False)
          liststr="/ ".join(seg_list)

           # 读取停用词文件
          f_stop = open(stopwords_path,'r', encoding='UTF-8')
          try:
            f_stop_text = f_stop.read()
          finally:
            f_stop.close()

          # 停用词清除
          f_stop_seg_list = f_stop_text.split('\n')
          for myword in liststr.split('/'):
            if not(myword.strip() in f_stop_seg_list) and len(myword.strip())>1:
              mywordlist.append(myword)

          docs_list.append(''.join(mywordlist))      # 存入文档列表中
          mywordlist=[]                  # 存入之后，需要清除mywordlist内容，防止重复
    finally:  
      file_object.close()

    print(f"docs_list:{docs_list}")

    docs = np.array(docs_list)
    print(f"docs:{docs}")
    
    
    
    print("-------------第一种方法-------------")
    count = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(count.fit_transform(docs)) 
    word=count.get_feature_names()#获取词袋模型中的所有词语
    weight1=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    print(weight1)
    for i in range(len(weight1)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print(u"-------这里输出第",i,u"类文本的词语tf-idf权重------")
        for j in range(len(word)):
            print(word[j],weight1[i][j])

 # 第二种方法是直接用TfidfVectorizer完成向量化与TF-IDF预处理
    print("-------------第二种方法-------------")
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf2 = TfidfVectorizer()
    re = tfidf2.fit_transform(docs)
    word=count.get_feature_names()#获取词袋模型中的所有词语
    weight2=re.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    print(weight2)
    for i in range(len(weight2)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print(u"-------这里输出第",i,u"类文本的词语tf-idf权重------")
        for j in range(len(word)):
            print(word[j],weight2[i][j])
            
    
    
    
```

**另一种**https://blog.csdn.net/a1240663993/article/details/88259978

参数介绍

可以发现sklearn中的tfidf使用十分简单，但是其功能却十分强大，接下来我们介绍几个常用的参数：

1、input：[‘file’,‘filename’,‘content’],默认为content也就是认为你的输入是一堆现成的句子（corpus），若是将其设置为file则是输入一个file对象，filename的话则是由sklearn后台读取文本再将其转化。

2、encoding and decoder_error：前者在输入为文件或是二进制文本时使用，也就是sklearn需要对输入进行解码的格式， 而后者是其搭配的参数，负责控制解码错误时的操作(默认为strict也就是错误时会报错，单个人一般设置为ignore，因为一般做的都是英文分类所以在ascii码之外的字符也是我不需要考虑的字符）

3、preprocessor: 这是一个函数输入口，我们可以将需要对文本做的预处理写成func然后放在这里，这样sklearn就会后台帮我们调用并对所有文本执行（很方便的说）

4、analyzer： [‘word’,‘char’,‘char_wb’] 控制我们是以字符级还是词级来提取n-gram特征，word为默认值，我们来讲讲char时的表现，sklearn会将一段英文拆分为一个个字母，然后根据设置的n-gram对其提取特征，至于char_wb则是对每个二进制字符提取特征（做代码分类时也许粒度最细的char_wb会有奇效哦）

5、stop_words：[{‘englist’}, None] 默认为None， 可以使用englist来删除英文中的常见停用词，也可以使用自定义的停用词库

6、max_features: int值，根据tf-idf值排序，取前max_features作为特征


[互信息](https://mp.weixin.qq.com/s/uIL1xytxW5GQyg3E1Y496w)

用得方法和我类似 以后可以参考https://blog.csdn.net/SMith7412/article/details/88298996
