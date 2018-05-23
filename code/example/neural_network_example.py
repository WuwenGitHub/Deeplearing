import tensorflow as tf
import numpy as np

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

#导入数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

#神经网络输入
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#搭建网络
#神经层 输入层+隐藏层+输出层
#输入层 1; 隐藏层 10; 输出层 1

#隐藏层
l1 = add_layer(xs, 1, 10, activation_funcation = tf.nn.relu)
#输出层
prediction = add_layer(l1, 10, 1, activation_funcation = None)

#计算误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#训练
for i in range(1000):
    sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict = {xs:x_data, ys:y_data}))