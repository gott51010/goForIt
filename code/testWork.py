import tensorflow as tf
# 上一行不报错说明安装这个库成功了

print('tensorflow Version is {}'.format(tf.__version__))
# 查看当前的tf版本 2.0.0以上o

gpu_ok = tf.test.is_gpu_available()
print("GPU is work ", gpu_ok)
