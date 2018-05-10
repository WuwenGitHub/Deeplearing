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
#预测的概率分布
y = tf.nn.softmax(tf.matmul(x,W) + b)

#实际分布(one-hot vector)
y_ =  tf.placeholder("float",[None,10])

#训练模型
#成本函数--交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

#训练 反向传播算法
#使用梯度下降算法最小化交叉熵  0.01学习速率
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化变量
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#开始训练
for i in range(1000):
    #随机训练(stochastic training)/随机梯度下降训练
    #随机抓取数据中的100个批处理数据点
    #替换之前的占位符来运行train_step
    #1.减少计算开销 2.最大化地学习到数据集的总体特性
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys})
    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
