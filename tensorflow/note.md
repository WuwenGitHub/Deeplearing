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
        </div></br>
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
        
# MNIST机器学习入门  
    一、导入数据集<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>注：</b>Anaconda中input_data.py位于tensorflow\.examples\.tutorials\.mnist文件夹下
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <ol>
<li>MNIST数据单元组成:一张包含手写数字的图片xs和一个对应的标签ys</li>
<li>60000行的训练数据集(mnist.train)
   <ol>
       <li>训练数据集的图片 mnist.train.images
          <ul>
              <li>形状为[60000,784]的张量</li>
              <li>第一维  图片索引</li>
              <li>第二维  图片像素点索引</li>
              <li>张量中元素  某张图片里的每个像素的强度值,0或1</li>
          </ul>
      </li>
      <li>训练数据集的标签 mnist.train.labels
          <ul>
              <li>[60000,10]的数字矩阵</li>
              <li>介于0到9的数字&nbsp;&nbsp;描述给定图片里表示的数字</li>
              <li>标签数据one-hot vectors 除某一维数字为1外,其余全为0</li>
              <li>数字n表示一个只有在第n维度(从0开始)数字为1的10维向量.例:标签0表示为([1,0,0,0,0,0,0,0,0,0,0])</li>
          </ul>
      </li>
   </ol>
</li>
<li>10000行的测试数据集(mnist.test)</li>
</ol>
   二、 Softmax回归(softmax regression)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 可用作给不同的对象分配概率
<ol>
  <li>对图片像素进行加权求和<br>
    对于给定的输入图片x代表是数字i的证据表示为:<br><image src="../image/mnist1.png"/></li>
  <li>使用softmax函数转换成概率<br><image src="../image/mnist4.png" /></li>
</ol>
   三、 训练模型<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <ol>
 <li>定义评估指标(成本/损失)--尽量最小化<br>成本函数"交叉熵"(cross-entropy)<br><image src=</li>
</ol>


