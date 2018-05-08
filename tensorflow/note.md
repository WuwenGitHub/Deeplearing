# Tensorflow函数变化
  * tf.sub() --> tf.subtract()
  * tf.mul() --> tf.multiply()
  * tf.types.float32 --> 为tf.float32
  * tf.pact() --> tf.stact()  
  <div align="center"><table><tr><th width="500"><b>旧</b></th><th width="500"><b>新</b></th></tr></table></div>

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
        <div align="center"> <table>
 <tr><th width="500"><b>数据集</b></th><th width="500"><b>目的</b></th></tr>
 <tr><td>data_sets.train</td><td>55000 组 图片和标签, 用于训练。</td></tr>
 <tr><td>data_sets.validation</td><td>5000 组 图片和标签, 用于迭代验证训练的准确性。</td></tr>
 <tr><td>data_sets.test</td><td>10000 组 图片和标签, 用于最终测试训练的准确性。</td></tr>
                             </table></div><br>
