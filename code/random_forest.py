"""随机森林"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


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
# 分训练集和测试集
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

# 交叉验证一下效果
# 数据集划分为N份 依次取每一份做测试集,每n-1份做测试集,多次模型训练以观测模型的稳定性

# rfc = RandomForestClassifier(n_estimators=25)
# rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)
#
# clf = DecisionTreeClassifier()
# clf_s = cross_val_score(rfc, wine.data, wine.target, cv=10)
#
# plt.plot(range(1, 11), rfc_s, label="RandomForest")
# plt.plot(range(1, 11), clf_s, label="Decision Tree")
# plt.legend()
# plt.show()


# 随机森林和决策时在十组交叉验证的效果对比
# rfc_l = []
# clf_l = []
#
# for i in range(10):
#     rfc = RandomForestClassifier(n_estimators=25)
#     if __name__ == '__main__':
#         rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
#     rfc_l.append(rfc_s)
#     clf = DecisionTreeClassifier()
#     clf_s = cross_val_score(clf, wine.data, wine.target, cv=10).mean()
#     clf_l.append(clf_s)
#
# plt.plot(range(1, 11), rfc_l, label="RandomForest")
# plt.plot(range(1, 11), clf_l, label="Decision Tree")
# plt.legend()
# plt.show()

# n_estimators 的学习曲线
# 这段代码会运行几分钟才能画出图 run前慎重
# superpa = []
# for i in range(200):
#     rfc = RandomForestClassifier(n_estimators=i+1, n_jobs=-1)
#     rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
#     superpa.append(rfc_s)
# print(max(superpa), superpa.index(max(superpa)))
# plt.figure(figsize=[20, 5])
# plt.plot(range(1, 201), superpa)
# plt.show()

# 随机森林的重要属性和api
# 随机森林的接口和决策树完全一致
# 4个常用 apply fit predict score
# 此外还有predict_proba 返回每个测试样本对应被分到每一类标签的概率
# 标签有几个分类就返回几个概率
# 二分类问题predict_proba数值大于0.5的被分到1 小于0.5的分到0

# 关于模型和调参
# 衡量模型在未知数据上的准确率的指标 就叫做泛化误差
# 泛化误差受模型结构复杂度影响 我们要寻找平衡点
# 而影响复杂度的参数 比如决策树,深度深,枝叶多就越复杂
# 随机森林是以复数的决策树为基础 是天生的复杂度高的模型
# 所以在对待树模型时 我们的目标是剪枝,减复杂度,防止过拟合
# 常用参数对模型的影响度
# n_estimators 不影响单个模型的复杂度_
# max_depth 默认最大深度,减小数值使模型简化
# max_features morenauto是特征总数的开平方 增大使模型复杂 减小使模型简单
