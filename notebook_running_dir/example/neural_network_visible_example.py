import tensorflow as tf
import numpy as np
#导入可视化模块
import matplotlib.pyplot as plt

#构建一个完整的神经网络--添加神经层,计算误差,训练步骤,判断是否在学习
#案例: y = W(x^2) + b

#构建添加神经层函数
#inputs 输入
#in_size
#out_size
#activation_funcation 激励函数
def add_layer(inputs, in_size, out_size, activation_funcation=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_funcation is None:
        return Wx_plus_b
    else:
        return activation_funcation(Wx_plus_b)

# Make up some real data
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# Define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#搭建网络
#神经层 输入层+隐藏层+输出层
#输入层 1; 隐藏层 10; 输出层 1

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_funcation = tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_funcation = None)

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#构建图形,使用散点图描述真实数据之间关系
#plot the real data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

#训练
for i in range(1000):
    # training
    sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})
        # plot the prediction
        line = ax.plot(x_data, prediction_value, 'r-', lw = 5)
        plt.pause(1)