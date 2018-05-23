import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data  * 0.1 + 0.3

#搭建模型
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

#计算误差
loss = tf.reduce_mean(tf.square(y - y_data))

#传播误差
#采用梯度下降算法
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#训练
##初始化所有变量
##must hava if define variable
init = tf.global_variables_initializer()

#开始训练
##初始化
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))