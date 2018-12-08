###装载预训练的模型

#%matplotlib inline
from matplotlib import pyplot as plt

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import Callback

n = 25000
r = 0.2
ratio = 0.2
batch_size = 32

#######################################
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
######################################

##include_top: whether to include the 3 fully-connected layers at the top of the network
##加载VGG16网络的权重
#将网络权重加载到所有的卷积层
#该网络将作为一个特征检测器来检测我们将要添加到全连接层的特征
#VGG16  拥有16个可以训练权重的层和1.4亿个参数
print('application_VGG16')
model = applications.VGG16(include_top = False, weights = 'imagenet')
datagen = ImageDataGenerator(rescale = 1./255)

print('得到特征表示')
##将图像传进网络来得到特征表示，这些特征表示将会作为神经网络分类器的输入
#Found 20000 images belonging to classes.
print('train_features')
generator = datagen.flow_from_directory('./data/train',
	target_size = (150, 150),
	batch_size = batch_size,
	class_mode = None,
	shuffle = False)
bottleneck_features_train = model.predict_generator(generator, int(n * (1 - ratio)) // batch_size)
np.save(open('./features/bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

#Found 5000 images belonging to 2 classes.
print('validation_features')
generator = datagen.flow_from_directory('./data/validation/',
	target_size = (150, 150),
	batch_size = batch_size,
	class_mode = None,
	shuffle = False)
bottleneck_features_validation = model.predict_generator(generator, int(n * ratio) // batch_size)
np.save('./features/bottleneck_features_validation.npy', bottleneck_features_validation)

#图像在传递到网络中时是有序传递的，所以我们可以很容易地为每张图片关联上标签
print('关联标签')
train_data = np.load('./features/bottleneck_features_train.npy')
train_labels = np.array([0] * (int((1 - ratio) * n) // 2) + [1] * (int((1 - ratio) * n) // 2))
validation_data = np.load('./features/bottleneck_features_validation.npy') 
validation_labels = np.array([0] * (int(ratio * n) // 2) + [1] * (int(ratio * n) // 2))

##设计一个小型的全连接神经网络，附加上从VGG16中抽取到的特征，将他作为CNN的分类部分
print('little_net')
model = Sequential() 
model.add(Flatten(input_shape=train_data.shape[1:])) 
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(256, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(optimizer='rmsprop', 
loss='binary_crossentropy', metrics=['accuracy']) 
fitted_model = model.fit(train_data, train_labels, 
	epochs=15, 
	batch_size=batch_size, 
	validation_data=(validation_data, validation_labels[:validation_data.shape[0]]), 
	verbose=0, 
	callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outer=False), history])

##在 15 个 epoch 后，模型就达到了 90.7% 的准确度。
##这个结果已经很好了，注意现在每个 epoch 在我自己的电脑上跑也仅需 1 分钟
print('pig1')
fig = plt.figure(figsize=(15, 5)) 
plt.plot(fitted_model.history['loss'], 'g', label="train losses") 
plt.plot(fitted_model.history['val_loss'], 'r', label="val losses") 
plt.grid(True) 
plt.title('Training loss vs. Validation loss - VGG16') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend()

print('pig2')
fig = plt.figure(figsize=(15, 5)) 
plt.plot(fitted_model.history['acc'], 'g', label="accuracy on train set") 
plt.plot(fitted_model.history['val_acc'], 'r', label="accuracy on validation sete") 
plt.grid(True) 
plt.title('Training Accuracy vs. Validation Accuracy - VGG16') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend()

print('End!!!!')