"""剪枝算法"""
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


# 决策树的过拟合(Overfitting)问题
# 决策树的所有节点覆盖的训练样本都是纯的 这样分类的话
# 连训练样本中的噪声也会被学习进去 造成结果是
# 对于训练样本而言表现良好 误差极低 但是对于测试数据表现并不好
# 为了让决策树有更好的泛化性 要进行剪枝修正

# sklearn 的剪枝策略参数
# max_depth 限制深度
# 适用于高纬度低样本量 决策树每多生长一层 对样本量的参数会增加倍
# 因此可以从=3 开始尝试 视效果决定是否追加深度


# min_samples_leaf限制子节点
# 当叶子节点少于给定数值的话 这个枝就会被砍掉
# 搭配max_depth 可以让回归模型平滑 并保证每个叶子的最小尺寸
# 这个参数过小会导致过拟合 过大会阻碍模型学习数据
# 一般样本充足的话 可以从=5开始尝试
# 在回归模型里可以避免低方差,过拟合的出现
# 在类别较少的分类模型中 一般可以从=1开始


# 和楼上是好基友的min_samples_split 限制分割
# 一个节点至少包含X个训练样本时 这个节点在被允许分支

# max_features
# 限制分枝时考虑的特征个数
# 超过这个个数的都被舍弃 方法比较暴力 是强行停止决策
# 所以 在不确定各个特征的重要性时 此参数会导致模型学习不足

# min_impurity_decrease
# 限制信息增益的大小 小于数值不发生分枝

wine = load_wine()
a = pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
# 表的最后一列是target
# print(a)

# wine 里还有几个属性是接下来要用的
# 特证名 feature_names
# 分类名 target_names

# 分训练集和测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)
# test_size参数是指0.3作为测试集 0.7作为训练集
# print(wine.data.shape)   # 原始数据是(178, 13) 178行13个标签
# print(Xtrain.shape)  # 训练集是(124, 13) 124行13个标签

test = []

# 开始愉快地建模吧!
# 实例化
# clf = tree.DecisionTreeClassifier(criterion="entropy")
# random_state 参数设置分支随机性
# splitter 参数设置随即选项 "best"优先选择更重要的特征分类  "random"分支更随机,树更深,训练集拟合度降低
for i in range(10):
    # 训练10次
    clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, max_depth=i+1)
    clf = clf.fit(Xtrain, Ytrain)
    score = clf.score(Xtest, Ytest)
    print(score)
    test.append(score)

plt.plot(range(1, 11), test, color="red", label="max_depth")
plt.legend()
plt.show()

# 从训练结果图来看 当层数达到3-4的时候 就达到稳定了,之后再加层数也没有意义了
# 这样就可以确定 max_depth 的最佳值
# 以此类推 每一个参数 都可以通过这样循环的方式试出来
