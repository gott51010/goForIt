# 用分类树预测泰坦尼克号的幸存者

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np


# 导入数据集 r是为了防止路径出错
data = pd.read_csv(r"E:\work\pycharm\goForIt\sklearn\titanic\train.csv")

# print(data.info())
# 看一眼info RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
# PassengerId    891 non-null int64
# Survived       891 non-null int64
# Pclass         891 non-null int64
# Name           891 non-null object
# Sex            891 non-null object
# Age            714 non-null float64
# SibSp          891 non-null int64
# Parch          891 non-null int64
# Ticket         891 non-null object
# Fare           891 non-null float64
# Cabin          204 non-null object
# Embarked       889 non-null object



# 不难发现特征里面有object 5 个 而分类树模型只能给数字 不能给对象 so如果需要保留的话必须手动转成数字
# 年龄 Age 714  Cabin 204 Embarked 889 都是不足891的 有缺失值需要处理


# 筛选特征
# data.drop(['Cabin', 'Name''Ticket'], inplace=True,axis=1)
# inplace=True 参数是让drop后处理的数据覆盖掉原有的数据 如果能确定这样处理不修改的话 可以加这个参数
# 不加这参数的话 需要再找个变量接收处理后的返回值
# axis=1 删除列(因为是以最外面大括号的维度)
data = data.drop(['Cabin', 'Name','Ticket'], axis=1)

# Age数据是714,有缺失值 但感觉这个特征点有用 所以要处理一下
# 由于是年龄 所以这里用平均数填充 视情况也有可能会用0,中位数 or 别的统计学知识去填补
data["Age"] = data["Age"].fillna(data["Age"].mean())


# print(data.info())    #打出来看一眼
# Data columns (total 9 columns):
# PassengerId    891 non-null int64
# Survived       891 non-null int64
# Pclass         891 non-null int64
# Sex            891 non-null object
# Age            891 non-null float64
# SibSp          891 non-null int64
# Parch          891 non-null int64
# Fare           891 non-null float64
# Embarked       889 non-null object


# Embarked  889 还是有缺失 但是只缺2行 ,所以大部分还是可用的,因此只删除缺这个特征的两行即可
# 删掉所有有NaN的行(默认axis=0)  传入=1的话可以删除列
data = data.dropna()


# print(data.info())
# 整体数据变成889行了

labels = data["Embarked"].unique().tolist()
data["Embarked"] = data["Embarked"].apply(lambda x: labels.index(x))
labels.index("S")


# loc['a':'b']#选取ab两行数据 loc[:,'one']#选取one列的数据 iloc函数基于索引位
data.loc[:, "Sex"] = (data["Sex"] == "male").astype("int")
# print(data.iloc[:, 3])

x = data.iloc[:,data.columns != "Survived"]
# data.columns != "Survived"  会生成一个布尔值的列表 所以是返回为true的列全都取出来

# y则相反 所有等于y的都取出
y = data.iloc[:,data.columns == "Survived"]

# 划分训练集

Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3)

# print(Xtrain.head(10))
# 打出来看一眼 由于是随机的索引乱掉了 所以手动整理一下 Xtrain.index = range(Xtrain.shape[0])
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])


# random_state 随机写个25
# 分类模型
# clf = DecisionTreeClassifier(random_state=25)
# clf = clf.fit(Xtrain, Ytrain)
# score = clf.score(Xtest,Ytest)

# print(score)
# 0.7490636704119851  一次的训练结果不理想

# 追加进行交叉验证的训练
# clf = DecisionTreeClassifier(random_state=25)
# score = cross_val_score(clf, x, y, cv=10).mean()

# print(score)
# 0.7469611848825333  10次均值也不理想 不升反降
tr = []
te = []

for i in range(10):
    clf = DecisionTreeClassifier(random_state=25
                                 ,max_depth=i+1
                                 # 第一次调参追加 "entropy" 发现测试集表现更好了
                                 ,criterion="entropy")
    clf = clf.fit(Xtrain, Ytrain)
    score_tr = clf.score(Xtrain,Ytrain)
    score_te = cross_val_score(clf, x, y, cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)


# plt.plot(range(1,11),tr,color="red",label="train")
# plt.plot(range(1,11),te,color="blue",label="test")
# plt.xticks(range(1,11))
# plt.legend()
# plt.show()


# 网格搜索:我们给出多个参数 网格搜搜进行枚举给出效果最好的组合
# 使用时要控制取值范围

clf = DecisionTreeClassifier(random_state=25)

# np.linspace(0,0.5,50) # 0 - 0.5 递增50个随机有序的数
gini_threholds = np.linspace(0,0.5,50)
# entropy_threholds = np.linspace(0,1,50)

# 参数是字典型 深度这样的有范围的数据组的话值是列表
parameters = {
    "criterion": ("gini", "entropy")
    , "splitter": ("best", "random")
    , "max_depth": [*range(2, 8)]
    , "min_samples_leaf": [*range(1, 20, 5)]
    # 信息增益最小值
    , "min_impurity_decrease": gini_threholds
}
# 网格搜索 1.要验证测模型 2.参数(字典) 3.次数
GS = GridSearchCV(clf, parameters, cv=10)
GS.fit(Xtrain, Ytrain)
print(GS.best_params_)

# {'criterion': 'entropy', 'max_depth': 6, 'min_impurity_decrease': 0.01020408163265306, 'min_samples_leaf': 1, 'splitter': 'best'}
