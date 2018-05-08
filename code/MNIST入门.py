#MNIST机器学习入门

#1. 导入数据集
#注: Anaconda中input_data.py位于tensorflow.examples.tutorials.mnist文件夹下
#60000行的训练数据集(mnist.train)
#Ⅰ训练数据集的图片 mnist.train.images
#        形状为[60000,784]的张量
#        第一维  图片索引
#        第二维  图片像素点索引
#        张量中元素  某张图片里的每个像素的强度值,0或1
#Ⅱ 训练数据集的标签 mnist.train.labels
#          [60000,10]的数字矩阵
#         介于0到9的数字   描述给定图片里表示的数字
#         标签数据one-hot vectors  除某一维数字为1外，其余全为0
#10000行的测试数据集(mnist.test)
#MNIST数据单元构成: 一张包含手写数字的图片(28x28像素)xs和一个对应的标签ys
#①将图片像素数组展开为一个向量(28x28=784维向量空间)
#②
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#softmax回归
#第一步  对图像像素值进行加权求和
#         不属于该类  权值为负
#         加入偏置量(bias)  消除无关干扰量
import tensorflow as tf

x = tf.placeholder("float",[None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ =  tf.placeholder("float",[None,10])
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