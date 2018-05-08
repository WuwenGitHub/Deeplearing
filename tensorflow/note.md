# Tensorflow函数变化
  <div align="center">
    <table>
      <tr><th width="500"><b>旧</b></th><th width="500"><b>新</b></th></tr>
      <tr><td>tf.sub()</td><td>tf.subtract()</td></tr>
      <tr><td>tf.mul()</td><td>tf.multipy()</td></tr>
      <tr><td>tf.types.float32</td><td>tf.float32</td></tr>
      <tr><td>tf.pact()</td><td>tf.stact()</td></tr>
    </table>
  </div></br>  
  
# MNIST
  * input\_data.py  
       下载用于训练和测试的MNIST数据集的源码  
  * maybe\_download()  
       将训练数据下载到本地文件夹中  
       文件夹名字由fully\_connected_feed.py文件顶部一个标记变量指定
  * extract\_images()、extract\_labels()函数  
       手动解压图片数据  
       形成2维的tensor[image index, pixel index]  
       "image index"/  数据集中图片编号  
       "pixel index"/    图片中像素点个数(0~图片像素上限)
  * 数据集对象  
        <div align="center">
          <table>
            <tr><th width="500"><b>数据集</b></th><th width="500"><b>目的</b></th></tr>
            <tr><td>data_sets.train</td><td>55000 组 图片和标签, 用于训练。</td></tr>
            <tr><td>data_sets.validation</td><td>5000 组 图片和标签, 用于迭代验证训练的准确性。</td></tr>
            <tr><td>data_sets.test</td><td>10000 组 图片和标签, 用于最终测试训练的准确性。</td></tr>
          </table>
        </div><br>
  * MNIST入门  
    <pre><code>from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
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
    </code></pre>
