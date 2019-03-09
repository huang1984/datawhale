
**1. 朴素贝叶斯的原理** 

**2. 利用朴素贝叶斯模型进行文本分类**  

**3. SVM的原理** 

**4. 利用SVM模型进行文本分类** 

**5. pLSA、共轭先验分布；LDA主题模型原理** 

**6. 使用LDA生成主题特征，在之前特征的基础上加入主题特征进行文本分类**

[sklearn：朴素贝叶斯（naïve beyes）](https://blog.csdn.net/u013710265/article/details/72780520)

[机器学习算法总结之朴素贝叶斯法](https://blog.csdn.net/Kaiyuan_sjtu/article/details/80030005)

**重点**：[基于scikit-learn的SVM实战](https://blog.csdn.net/Kaiyuan_sjtu/article/details/80064145)

**利用SVM模型进行文本分类**


```python
from sklearn.datasets import load_iris
X = load_iris().data
y = load_iris().target
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit

cv_split = ShuffleSplit(n_splits=5, train_size=0.7, test_size=0.25)
for train_index, test_index in cv_split.split(X):
    train_X = X[train_index]
    test_X = X[test_index]
    train_y = y[train_index]
    test_y = y[test_index]
    svc_model = SVC()
    svc_model.fit(train_X, train_y)
    score = svc_model.score(test_X, test_y)
    print(score)
```

多个方法的集合：[手把手教你在Python中实现文本分类（附代码、数据集）](https://blog.csdn.net/sinat_38682860/article/details/80421697)


```python
from sklearn import model_selection, preprocessing, svm, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
#加载数据集
data = open('data/corpus').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
content = line.split()
labels.append(content[0])
texts.append(content[1])
 
#创建一个dataframe，列名为text和label
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
 
# label编码为目标变量
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
#创建一个向量计数器对象
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])
 
#使用向量计数器对象转换训练集和验证集
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

#训练主题模型
lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_
vocab = count_vect.get_feature_names()
 
#可视化主题模型
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
topic_summaries.append(' '.join(topic_words)
                       
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
# fit the training dataset on the classifier
classifier.fit(feature_vector_train, label)
 
# predict the labels on validation dataset
predictions = classifier.predict(feature_vector_valid)
 
if is_neural_net:
predictions = predictions.argmax(axis=-1)
 
return metrics.accuracy_score(predictions, valid_y)
                       
ccuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print "SVM, N-Gram Vectors: ", accuracy                       
                       
```

[用LDA处理文本(Python)](https://blog.csdn.net/u013710265/article/details/73480332)

[LDA主题模型笔记](https://blog.csdn.net/Kaiyuan_sjtu/article/details/83572927)
