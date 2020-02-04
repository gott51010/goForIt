"""另一套数据的随机森林测试 sklearn 自带的breast_cancer"""
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取数据
data = load_breast_cancer()

# 数据全体
# print(data)

# (569, 30)
# 本套数据的特征是 结构清晰简单 569条记录 30个特征
# 训练难点在于 样本量小 容易过拟合
# print(data.data.shape)

# print(data.target)

# rfc = RandomForestClassifier(n_estimators=100, random_state=90)
# score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean()
# print(score_pre)


# 调试模型 先拿n_estimators 开刀
# 看一下学习曲线的趋势 观察n_estimators 在什么取值范围变得平滑
# 取10个数为一个阶段 观察模型准确率的变化

# scorel = []
# for i in range(0, 200, 10):
#     rfc = RandomForestClassifier(n_estimators=i+1,
#                                  n_jobs=-1,
#                                  random_state=90)
#     score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
#     scorel.append(score)
# print(max(scorel), (scorel.index(max(scorel)) * 10) + 1)
# plt.figure(figsize=[20, 5])
# plt.plot(range(1, 201, 10), scorel)
# plt.show()

"""
观察趋势 从曲线跑出的结果中选取一个更小区间 再跑曲线
"""

# scorel = []
# for i in range(60, 99):
#     rfc = RandomForestClassifier(n_estimators=i,
#                                  n_jobs=-1,
#                                  random_state=90)
#     score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
#     scorel.append(score)
# print(max(scorel), ([*range(60, 99)][scorel.index(max(scorel))]))
# plt.figure(figsize=[20, 5])
# plt.plot(range(60, 99), scorel)
# plt.show()

# 运行结果 0.9631265664160402 73
# 可以看到这个区间内 表现最好的参数取值是73

"""
调整max_depth参数
调参最好一个一个调整,控制变量 根据数据大小尝试范围
"""
# # 对于大型数据可以尝试30-50层的深度 本次数据很小 可以尝试1-10
# param_grid = {'max_depth': np.arange(1, 20, 1)}
# # n_estimators 就用上一步调整出的最佳值 73
# rfc = RandomForestClassifier(n_estimators=73, random_state=90)
# score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean()
# print(score_pre)    # 本次运行结果 0.9631265664160402
# GS = GridSearchCV(rfc, param_grid, cv=10)
# GS.fit(data.data, data.target)
# print(GS.best_params_)  # 本次运行结果 {'max_depth': 8}
# print(GS.best_score_)   # 本次运行结果 0.9666666666666666

# 因为每次的运行结果会有误差
# 如果发现限制 max_depth 之后准确率降低的话
# 说如果限制此参数会导致拟合不足


"""
调整max_features参数
max_features 即可以减小拟合度也可以增大拟合度
调整此参数之前应当先判断模型目前的状态
嘉定刚才的max_depth值的调整已经使模型过于简单 现在希望它加大拟合度
那就应当把参数往大了调整
"""
# param_grid = {'max_features': np.arange(5, 30, 1)}
# # n_estimators 就用上一步调整出的最佳值 73
# rfc = RandomForestClassifier(n_estimators=73, random_state=90)
# GS = GridSearchCV(rfc, param_grid, cv=10)
# GS.fit(data.data, data.target)
# print(GS.best_params_)  # 本次运行结果 {'max_features': 24}
# print(GS.best_score_)   # 本次运行结果 0.9666666666666668


"""
调整min_sample_leaf参数
本次数据量小 采用从最小值开始+10 
高纬度数据的话可能会有+50甚至更多的情况 可以多做尝试
"""
# param_grid = {'min_samples_leaf': np.arange(1, 1+20, 1)}
# # n_estimators 就用上一步调整出的最佳值 73
# rfc = RandomForestClassifier(n_estimators=73, random_state=90)
# GS = GridSearchCV(rfc, param_grid, cv=10)
# GS.fit(data.data, data.target)
# print(GS.best_params_)  # 本次运行结果 {'min_samples_leaf': 1}
# print(GS.best_score_)   # 本次运行结果 0.9666353383458647


"""
调整min_samples_split参数
"""
# param_grid = {'min_samples_split': np.arange(2, 2+20, 1)}
# # n_estimators 就用上一步调整出的最佳值 73
# rfc = RandomForestClassifier(n_estimators=73, random_state=90)
# GS = GridSearchCV(rfc, param_grid, cv=10)
# GS.fit(data.data, data.target)
# print(GS.best_params_)  # 本次运行结果 {'min_samples_split': 2}
# print(GS.best_score_)   # 本次运行结果 0.9666353383458647


"""
调整criterion参数
"""
# 注意这里设置参数范围的写法和别的不一样
# param_grid = {'criterion': ['gini', 'entropy']}
# # n_estimators 就用上一步调整出的最佳值 73
# rfc = RandomForestClassifier(n_estimators=73, random_state=90)
# GS = GridSearchCV(rfc, param_grid, cv=10)
# GS.fit(data.data, data.target)
# print(GS.best_params_)  # 本次运行结果 {'criterion': 'gini'}
# print(GS.best_score_)   # 本次运行结果 0.9666353383458647


# 我们看到 调整到后期 模型并没有什么进步空间
# 说明剩余的误差是数据噪声决定的
# 无法通过方差偏差来决定 不能再通过调参来调整了
# 如果实际中还需要继续精确 可以通过换算法,数据预处理


"""
调整结束 总结一下调整结果
"""
# 无调整的参数
rfc = RandomForestClassifier(n_estimators=100, random_state=90)
score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean()
print(score_pre)    # 0.9648809523809524

# 调整后
rfc = RandomForestClassifier(n_estimators=73, random_state=90)
score = cross_val_score(rfc, data.data, data.target, cv=10).mean()

print(score)    # 0.9666353383458647

print(score - score_pre)    # 0.0017543859649122862
