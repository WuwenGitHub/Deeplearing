### 装载Keras及其依赖包，进行处理

#%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMNotebookCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback


### 图像生成器和数据增强
## 数据增强(data augmentation)
# 生成更多图片，防止过拟合，有助于模型保持更好的泛化性
batch_size = 32

## ImageDataGenerator类 无限制地从训练集和测试集中批量引入图像流
# 于ImageDataGenerator中，对每个批次引入随机修改 对图片进行缩放
'''
ImageDataGenerator(
      ....
      rescale=None,   //重放缩因子，默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
      shear_range=0.,//浮点数，剪切强度(逆时针方向的剪切变换角度)
      zoom_range=0.,//浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
      horizontal_flip=False,//布尔值，进行随机水平翻转
      ....
)
'''
train_datagen = ImageDataGenerator(rescale=1/255.,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1/255.)

# 创建两个文件生成器
'''
ImageDataGenerator.flow_from_directory(...):
以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据

ImageDataGenerator.flow_from_directory(
      directory,                      //目标文件夹路径.对于每一个类，该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、NP、PPM的图片都会被生成器使用.
      target_size,                  //整数tuple，默认为(256,256).图像将被resize成该尺寸
      color_mode,                //颜色模式,为"grayscale","rgb"之一,默认为"rgb".代表这些图片是否会被转换为单通道或三通道的图片.
      classes,                       //可选参数,为子文件夹的列表,如['dogs','cats']默认为None. 若未提供,则该类别列表将从directory下的子文件夹名称/结构自动推断。每一个子文件夹都会被认为是一个新的类。(类别的顺序将按照字母表顺序映射到标签值)。通过属性class_indices可获得文件夹名与类的序号的对应字典。
      class_mode,                //"categorical", "binary", "sparse"或None之一. 默认为"categorical. 该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签,"binary"返回1D的二值标签."sparse"返回1D的整数标签,如果为None则不返回任何标签, 生成器将仅仅生成batch数据, 这种情况在使用model.predict_generator()和model.evaluate_generator()等函数时会用到.
      batch_size,                  //batch数据的大小，默认32
      shuffle,                          //是否打乱数据,默认为True
      seed,                             //可选参数，打乱数据和进行变换时的随机数种子
      save_to_dir,                 //None或字符串,该参数能让你将提升后的图片保存起来,用以可视化
      save_prefix,                 //字符串,保存提升后图片时使用的前缀,仅当设置了save_to_dir时生效
      save_format,                //"png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"
      flollow_links                  //是否访问子文件夹中的软链接
)
'''
train_generator = train_datagen.flow_from_directory('./data/train/',
                                                   target_size=(150, 150),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

validation_generator = val_datagen.flow_from_directory('./data/validation/',
                                                   target_size=(150, 150),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

### 模型结构
# 使用拥有3个卷积/池化层和2个全连接层的CNN
model = Sequential()

## 3个卷积/池化层
# 分别使用32,32,64的3*3滤波器

# model.add(...)  用于堆叠模型
'''
keras.layers.convolutional.Conv2D(...)
    二维卷积层，即对图像的空域卷积。该层对二维输入进行滑动窗卷积，
    当使用该层作为第一层时，应提供input_shape参数。
    例如input_shape = (128,128,3)代表128*128的彩色RGB图像（data_format='channels_last'）
keras.layers.convolutional.Conv2D(
  filters, 
  kernel_size, 
  strides=(1, 1), 
  padding='valid', 
  data_format=None, 
  dilation_rate=(1, 1), 
  activation=None, 
  use_bias=True, 
  kernel_initializer='glorot_uniform', 
  bias_initializer='zeros', 
  kernel_regularizer=None, 
  bias_regularizer=None, 
  activity_regularizer=None, 
  kernel_constraint=None, 
  bias_constraint=None)
'''
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), padding='same', activation='relu'))

'''
为空域信号施加最大值池化
keras.layers.convolutional.MaxPooling2D(
  pool_size=(2, 2), 
  strides=None, 
  border_mode='valid', 
  dim_ordering='th')
'''

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'), activation='relu')
model.add(MaxPooling2D(pool_size=(2, 2)))

## 2个全连接层的CNN
# 使用dropout来避免过拟合
'''
keras.layers.core.Dropout(rate, noise_shape=None, seed=None)
为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按
一定概率(rate)随机断开输入神经元，Dropout层用于防止过拟合。
'''
model.add(Dropout(0.25))
'''
keras.layers.core.Flatten()
Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到
全连接层的过渡。Flatten不影响batch的大小。
'''
model.add(Flatten())
'''
keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
  kernel_constraint=None, bias_constraint=None)
  Dense就是常用的全连接层，所实现的运算是output = activation(dot(input, kernel)+bias)。
  其中activation是逐元素计算的激活函数，kernel是本层的权值矩阵，bias为偏置向量，
  只有当use_bias=True才会添加。
'''
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

## 使用随机梯度下降法进行优化
# 参数
# learning rate  0.01
# momentum       0.9
epochs = 50
lrate = 0.01
decay = lrate/epochs
'''
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
随机梯度下降法，支持动量参数，支持学习衰减率，支持Nesterov动量
'''
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

## 检查模型
model.summary()

## 训练前，定义两个于训练时调用的回调函数

# Callback for loss logging per epoch
# 存储每个时期的损失和精确度指标 用于绘制训练错误图表
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
history = LossHistory()

# Callback for early stopping the training
# 损失函数无法改进在测数据的效果时，停止训练
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0,
                                              patience=2,
                                              verbose=0,
                                              mode='auto')

#使用keras_tqdm,监视模型的训练过程
n = 25000
ratio = 0.2

#使用fit_generator方法  将生成器作为输入的变体(标准拟合方法)
#训练模型的时间超过50的epoch
fitted_model = model.fit_generator(train_generator,
                                  steps_per_epoch = int(n * (1 - ratio)) //batch_size,
                                  epochs = 50,
                                  validation_data = validation_generator,
                                  validation_steps = int(n * ratio) //batch_size,
                                  callbacks = [TQDMNotebookCallback(leave_inner = True, leave_outer = True), early_stopping, history], 
                                  verbose = 0)

#保存结果
print('保存结果')
model.save('./models/model4.h5')

#绘制训练和测试中损失指标值
print('绘制训练和测试中损失指标值')
losses, val_losses = history.losses, history.val_losses
fig = plt.figure(figsize=(15, 5))
plt.plot(fitted_model.history['loss'], 'g', label="train losses") 
plt.plot(fitted_model.history['val_loss'], 'r', label="val losses")
plt.grid(True)
plt.title('Training loss vs. Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#绘制训练集和测试集上的准确度
print('绘制训练集和测试集上的准确度')
losses, val_losses = history.losses, history.val_losses
fig = plt.figure(figsize=(15, 5))
plt.plot(fitted_model.history['acc'], 'g', label="accuracy on train losses")
plt.plot(fitted_model.history['val_acc'], 'r', label="accuracy on validation losses")
plt.grid(True)
plt.title('Training Accuracy vs. Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

print('END!!!')