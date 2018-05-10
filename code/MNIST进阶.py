#初始化时加入少量的噪声来打破对称性以及避免0梯度
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#使用ReLU神经元--用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#卷积使用1步长(stride size),0边距(padding size)模板--保证输入和输出是同一个大小
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化用简单的2x2大小的模板
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#第一层卷积
#由一个卷积接一个max pooling完成

#卷积权重张量形状[5, 5, 1, 32]
#第一、二维 patch的大小
#第三维 输入的通道数目
#第四维 输出的通道数目
W_conv1 = weight_variable([5, 5, 1, 32])
#一个输出通道一个偏置量
b_conv1 = bias_variable([32])
#将x变为一个4d向量
#第二、三维 图片的宽、高
#第四维 图片的颜色通道(灰度--1 rgb--3)
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#地二层卷积
#每5x5的patch会得到64个特征
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
#图片处理--将图片尺寸减小到7x7 加上一个1024个神经元的全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#将池化层输出的张量reshape成向量,乘上权重矩阵,加上偏置,然后对其使用ReLU
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
#减少过拟合
#使用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
#softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#训练和评估模型
#使用ADAM优化器来做梯度最速下降
#于feed_dict中加入额外的参数keep_prob来控制dropout比例
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
