"""数据预处理
"""

# 数据预处理 把不准确或不适用于模型记录的数据纠正或删除掉
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
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


# preprocessing.StandaradScaler
# 标准化
# 当数据(x)按均值中心化后,在按标准差(o)缩放 数据就会服从为均值为0 方差为1的正态分布
# 这个过程叫做 数据标准化 Z-score normalization

# from sklearn.preprocessing import StandardScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

scaler = StandardScaler()
# fit 的本质是生成均值和方差
scaler.fit(data)
# 均值属性
print(scaler.mean_)
# 方差属性
print(scaler.var_)
# 导出结果
x_std = scaler.transform(data)
print(x_std.mean())
print(x_std.std())

# fit_transform 两步合一步
scaler.fit_transform(data)
# 逆标准化
scaler.inverse_transform(data)


# 注意点:
# StandardScaler 和 MinMaxScaler  控制NaN会被当做是缺失值,在fit的时候忽略,在transform的时候保持缺失NaN的状态显示
# 并且fit接口中 只允许导入至少二维数组 一位数组会报错
# 嘛..不过实际运用中输入的X会是特征矩阵,矩阵不可能是一位数组 所以几乎不存在这个问题


# 缺失值的处理
# 数据不是完美的 所以对于建模来说需要填充缺失值

from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer()
# 参数指定 用中位数填补
imp_median = SimpleImputer(strategy="median")
# 用0 填补
imp_median = SimpleImputer(strategy="constant", fill_value=0)


# 除了这个库以外 也可用panda 和 Numpy
import pandas as pd
# 下面的代买因为没有合适的csvdata 所以跑不起来...
# data = pd.read_csv()      #准备csv 假装这里有数据
# data.head()
# data.lot[:, "number"] = data.loc[:, "number"].fillna(data.loc[:, "number"].median())
# # .fillna 在 DataFrame里面直接进行填充
#
# data.dropna(axis=0, inplace=True)


# 分类型特征的处理
# 编码与哑变量
# 除了专门用来处理文字的算法,其他算法在fit的时候基本要求是传入数组矩阵
# 但是现实生活中 许多特征收集完的时候是通过文字表现得
# 这种情况下为了适应算法 必须对文字进行编码 把文字转化成数字

# preprocessing.LabelEncoder
# 标签专用,能够将分类转化为分类数值

from sklearn.preprocessing import LabelEncoder

# 也是没准备数据跑不了...
# 这里要输入的是标签 不是特征矩阵
# y = data.iloc[:, -1]
# le = LabelEncoder()
# le = le.fit(y)
# label = le.transform(y)
#
# 属性 classes_ 查看标签中有多少类别
# print(le.classes_)
# 查看结果
# print(label)
#
# le.fit_transform(y)
# le.inverse_transform(label)
# 赋值回去
# data.iloc[:, -1] = label
# data.head()


# 上述可以偷懒为一行的写法
# from sklearn.preprocessing import LabelEncoder
# data.iloc[:, -1] = LabelEncoder().fit_transform(data.iloc[:, -1])

