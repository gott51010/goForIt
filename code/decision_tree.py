"""决策树"""
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import pydotplus
import graphviz


wine = load_wine()
a = pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
# 表的最后一列是target
print(a)


# wine 里还有几个属性是接下来要用的
# 特证名 feature_names
# 分类名 target_names
#
# 分训练集和测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)
# test_size参数是指0.3作为测试集 0.7作为训练集
print(wine.data.shape)   # 原始数据是(178, 13) 178行13个标签
print(Xtrain.shape)  # 训练集是(124, 13) 124行13个标签


# 开始愉快地建模吧!
# 实例化
# clf = tree.DecisionTreeClassifier(criterion="entropy")
# random_state 参数设置分支随机性
# splitter 参数设置随即选项 "best"优先选择更重要的特征分类  "random"分支更随机,树更深,训练集拟合度降低
clf = tree.DecisionTreeClassifier(criterion="entropy")
# fit是训练接口
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest)  # 注意参数顺序
print(score)  # 预测精确度  我这次跑是0.9629629629629

# 下面参数的name可以换成中文 然而显示不好pass了
feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']
# 更直观的数据 决策树可视化
# 里面放入刚建好的模型
dot_data = tree.export_graphviz(clf, feature_names=wine.feature_names,
                                class_names=["beer", "juice", "water"],
                                filled=True,    # 分类颜色 以及颜色越浅当前分类不纯度越高
                                rounded=True    # 圆角方形
                                )

# 跑不了这个库的话 需要安装Graphviz
graph = pydotplus.graph_from_dot_data(dot_data)
# 利用pydotplus 把决策树导出为图片(pdf也可)
with open('../LogisticTest/tree_graph.png', 'wb') as f:
    f.write(graph.create_png())

# 特征重要性 feature_importances_
print(clf.feature_importances_)
# 可见特征里只有4项是有数字的 说明对于构造这个决策树是只有这四项是有重要意义的
"""
[0.10151147 0.         0.         0.         0.         0.
 0.41155238 0.         0.         0.18719674 0.         0.
 0.29973942]
"""
# zip 关联特征和重要性
show_zip = {*zip(wine.feature_names, clf.feature_importances_)}
print(show_zip)


