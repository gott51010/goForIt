""" 手写数据 """
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


digits = datasets.load_digits()

for label, img in zip(digits.target[:10], digits.images[:10]):
    # 两行5列显示
    plt.subplot(2,5,label + 1)
    plt.axis('off')
    # 显示图片
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    # 显示标题
    plt.title('Digit: {0}'.format(label))

plt.show()

# 读取3和8的位置
flag_3_8 = (digits.target == 3) + (digits.target == 8)

# 3和8的图片 and 标签
images = digits.images[flag_3_8]
labels = digits.target[flag_3_8]

# 降维
images = images.reshape(images.shape[0], -1)

# 分類器
n_samples = len(flag_3_8[flag_3_8])
train_size = int(n_samples * 3 / 5)
# 生成分类器
classifier = tree.DecisionTreeClassifier()
# 给与学习data
classifier.fit(images[:train_size], labels[:train_size])

# 结果正确的测试数据
expected = labels[train_size:]
# 分类结果 预测正确的数据
predicted = classifier.predict(images[train_size:])

# 算出正确率
print('Accuracy:\n',
      accuracy_score(expected, predicted))

# 混同行列(真伪表)
# 例: [[59 16]
#  [ 2 66]]
# 即:
#         预测3  预测8
# 实际3   59      16
# 实际8    2      66
#

print('Confusion matrix:\n',
      confusion_matrix(expected, predicted))

# 适合率 是由预测率中为真的比例算出的
print('Precision:\n',
      precision_score(expected, predicted, pos_label=3))
# 再现率
print('Recall:\n',
      recall_score(expected, predicted, pos_label=3))
# F值
print('F-measure:\n',
      f1_score(expected, predicted, pos_label=3))
