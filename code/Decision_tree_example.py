#  一维回归树案例
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# 创建一组有噪声的曲线 x随机取值0-5 用正弦函数sin去生成y值
# 但是人为加入一些y的噪声

# 生成随机数种子
rng = np.random.RandomState(1)
# np.random.rand 生成随机数据的函数
# 这里因为已经直接声明rng了 所以可以直接写rand 生成0-1之间的随机数
# rng.rand(5) 输入5的话会返回5个随机数
# print(rng.rand(10,7))  输入数字结构的话 会返回10,7 形状的随机数数组

# 这里的rand(80, 1) 生成了数组是因为 tree接口是不允许导入一维特征的
# 不在生成数据的时候整形的话 之后还要再调用reshape升维 so还不如现在一步到位
# 乘5 是因为设计时候希望是0-5来着 扩大一下 再sort正序一下
x = np.sort(5 * rng.rand(80, 1), axis=0)

# 由于导入的X是二维的 所以Y也会是二维
# 但y如果是二维的话会报错 于是只能用ravel 手动降维
y = np.sin(x).ravel()
# 手动加点噪声
y[::5] += 3 * (0.5 - rng.rand(16))

# # 画图瞄一眼
# plt.figure()
# # 这个是画散点图的工具
# plt.scatter(x, y, s=20, edgecolors="black", c="darkorange", label="data")
# plt.show()

# 实例化 and训练 建两个深度对比效果
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(x,y)
regr_2.fit(x,y)

# 生成测试集验证
# arange(开始点,结束点,步长) 生成有序数列 后面加切片是为了增维
x_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

# 分别导入验证
y_1 = regr_1.predict(x_test)
y_2 = regr_2.predict(x_test)


# 画图瞄一眼
plt.figure()
# 这个是画散点图的工具
plt.scatter(x, y, s=50, edgecolors="black", c="darkorange", label="data")
plt.plot(x_test, y_1, color="blue", label="max_depth=2", linewidth=2)
plt.plot(x_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decison tree")
# 显示图例
plt.legend()
plt.show()

# 结果小结 深度2还比较准确 深度5过拟合了


# 补充另一个豆知识
# np的矩阵的维度  最外面的括号代表着 axis=0，依次往里的括号对应的 axis 的计数就依次加 1
# test = np.array([[1, 2], [3, 4]])
# a = np.sum(test, axis=0)
# b = np.sum(test, axis=1)
# print(a)
# print(b)
