"""数据预处理
"""

# 数据预处理 把不准确或不适用于模型记录的数据纠正或删除掉
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


# 无量纲化，也称为数据的规范化，消除量纲影响后再进行接下来的分析。
# 在梯度和矩阵为核心的算法中 无量纲化可以加快求解速度
# 有一个特例是决策树和树的集成算法里 不需要无量纲化 决策树可以处理任意数据
# 线性的无量纲化包括去中心化处理 和缩放处理等
# 中心化处理 让所有记录减去一个固定值 让样本平移到特定位置
# 缩放处理 通过除以一个固定值 将数据固定在某个范围里


# sklearn的preprocessing.MinMaxScaler()
# 数据按照最小值中心化后,在按照极差(最大值,最小值)缩放
# 即:数据移动了最小值个单位,并且收敛到[0,1]之间
# 这个过程叫做数据归一化(Min-Max Scaling)


data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
# print(pd.DataFrame(data))
# 对应索引 四行两列的表
#     0   1
# 0 -1.0   2
# 1 -0.5   6
# 2  0.0  10
# 3  1.0  18



# 实现归一化
scaler = MinMaxScaler()
# fit 本质是生成min(x)和max(x)
scaler = scaler.fit(data)
# 通过接口导出结果
result = scaler.transform(data)
# 可以看到 result的范围已经被压缩到[0,1]了
# [[0.   0.  ]
#  [0.25 0.25]
#  [0.5  0.5 ]
#  [1.   1.  ]]

# 训练和导出 fit_transform一步到位
result_ = scaler.fit_transform(data)    # 结果同result

# 反向操作 返回归一化之前的原数据
scaler.inverse_transform(result)
# print(scaler.inverse_transform(result))
# [[-1.   2. ]
#  [-0.5  6. ]
#  [ 0.  10. ]
#  [ 1.  18. ]]

# 参数feature_range
# 使用MinMaxScaler 默认数据归一化到[0,1]的范围中
# 参数feature_range可以将数据归一化到其他范围


scaler = MinMaxScaler(feature_range=[5, 10])
result_2 = scaler.fit_transform(data)
# print(result_2)
# [[ 5.    5.  ]
#  [ 6.25  6.25]
#  [ 7.5   7.5 ]
#  [10.   10.  ]]

# 备胎接口 partial_fit()
# 当x中特征数量非常多的时候,fit会报错 此时就可以用备胎了

# 归一化除了专业的sklearn之外 numpy也是可以满足需求的
x = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
# 归一化
x_nor = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
print(x_nor)

# 逆归一化
x_return = x_nor*(x.max(axis=0) - x.min(axis=0)) + x.min(axis=0)
print(x_return)
