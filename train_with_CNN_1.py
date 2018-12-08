### 装载Keras及其依赖包，进行处理

%matplotlib inline
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

train_datagen = ImageDataGenerator(rescale=1/255.,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1/255.)


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
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

## 2个全连接层的CNN
# 使用dropout来避免过拟合
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#Dense(n, activation='sigmoid')  n--类别个数
#model.add(Dense(3, activation='softmax'))
model.add(Dense(2, activation='softmax'))

## 使用随机梯度下降法进行优化
# 参数
# learning rate  0.01
# momentum       0.9
epochs = 50
lrate = 0.01
decay = lrate/epochs
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
#n = 25872
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