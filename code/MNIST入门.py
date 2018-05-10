#MNIST机器学习入门

#1. 导入数据集
#注: Anaconda中input_data.py位于tensorflow.examples.tutorials.mnist文件夹下
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

#softmax回归
#第一步  对图像像素值进行加权求和
#         不属于该类  权值为负
#         加入偏置量(bias)  消除无关干扰量
# x--图片i  W--权重  b--偏置量(bias)
# x 非特定值 可输入
x = tf.placeholder("float",[None, 784])
#使用全为0的张量来初始化W,b 可输入
#需要学习,初始值可以随意设置
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#第二步  使用softmax函数将其转换成概率y
#softmax模型实现
y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ =  tf.placeholder("float",[None,10])
#训练模型 成本函数--交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys})
    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
