"""随机森林"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import graphviz


# 随机森林是非常有代表性的bagging集成算法 他的所有基评估器都是决策树
# 分类树组成的森林就叫随机森林分类器
# 回归树....................回归器

# 单个的决策树准确率越高 森林的准确率也会越高

# 随机森林里的参数
# n_estimators 森林里基数评估器的数量(即 树的数量)
# 当n_estimators达到一定程度后 森林的精准度也开始波动
# 这个参数越大 需要的计算量和内存也越大 训练时间越长

# 开始愉快地建个森林吧!
wine = load_wine()
#  分训练集和测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)

# 生成一棵树
clf = DecisionTreeClassifier(random_state=0)
# 生成森林
rfc = RandomForestClassifier(random_state=0)
# 注意! random_state=0 指定 也不能保证每次结果相同
# 因为DecisionTreeClassifier random_state=0 是生成同一棵树
# RandomForestClassifier random_state=0 是生成同一片森林 里面的每棵树是长得不一样的

clf = clf.fit(Xtrain, Ytrain)
rfc = rfc.fit(Xtrain, Ytrain)

score_c = clf.score(Xtest, Ytest)
score_r = rfc.score(Xtest, Ytest)

# print(score_c)    # 本次0.8888888888888888
# print(score_r)    # 本次0.9629629629629629
# 每次运行结果不一样,但是多跑几次差不多可以看出 森林的准确度一般大于等于单棵决策树




