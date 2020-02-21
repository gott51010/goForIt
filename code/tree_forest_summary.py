"""决策树与随机森林的小结"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# 森林中任意两棵树的相关性：相关性越大，错误率越大
# 随机森林的优缺点对比
# 超硬核的优点!
# 可以处理高维度(特征feature多)的数据,且特征子集是随机选择的,因此不用做特征选择
# 训练之后可给出哪些特征比较重要,检测feature的相互影响
# 对于不平衡的数据集来说,可以平衡误差,有特征遗失的情况下仍然可以维持精度

# 缺点:
# 对于噪声过大的数据会过拟合
# 对于有不同取值的属性的数据，取值划分较多的属性会对随机森林产生更大的影响，所以随机森林在这种数据上产出的属性权值是不可信的。
