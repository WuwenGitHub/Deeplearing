import tensorflow as tf

#placeholder 占位作用,运行时才进行传值
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    #placeholder需要传入的值放于feed_dict={}中
    #与placeholder一一对应
    #placeholder与feed_dict={}绑定出现
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))