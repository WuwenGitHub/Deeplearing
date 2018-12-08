### 将训练数据集按名字放入不同文件夹
## 文件结构
#data/
#  train/
#    cats/
#      cat001.jpg
#      cat002.jpg
#      cat003.jpg
#      ....
#    dogs/
#      dog001.jpg
#      dog002.jpg
#      dog003.jpg
#      ....
#  validation/
#    cats/
#      cat001.jpg
#      cat002.jpg
#      cat003.jpg
#      ....
#    dogs/
#      dog001.jpg
#      dog002.jpg
#      dog003.jpg
#      ....

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm_notebook
from random import shuffle
import shutil
import pandas as pd

def organize_datasets(path_to_data, n=4000, ratio=0.2):
    #返回path指定的文件夹包含的文件或文件夹的名字的列表
    files = os.listdir(path_to_data)
    #将多个路径组合后返回
    #将目录和文件名合成一个路径
    files = [os.path.join(path_to_data, f) for f in files]
    #将序列的所有元素随机排序
    shuffle(files)
    files = files[:n]
    
    n = int(len(files) * ratio)
    val, train = files[:n], files[n:]
    
    #递归删除目录及文件
    shutil.rmtree('./data/')
    print('/data/removed')
    
    for c in ['dogs', 'cats']:
        os.makedirs('./data/train/{0}'.format(c))
        os.makedirs('./data/validation/{0}'.format(c))


    print('folders created!')
    
    for t in tqdm_notebook(train):
        if 'cat' in t:
        	shutil.copy2(t, os.path.join('.', 'data', 'train', 'cats'))
        
        elif 'dog' in t:
        	shutil.copy2(t, os.path.join('.', 'data', 'train', 'dogs'))
        else:
            print('Other data')

    for t in tqdm_notebook(val):
        if 'cat' in t:
            shutil.copy2(t, os.path.join('.', 'data', 'validation', 'cats'))
        
        elif 'dog' in t:
            shutil.copy2(t, os.path.join('.', 'data', 'validation', 'dogs'))
        else:
            print('Other data')

            
    print('Data copied!')

n = 25000
ratio = 0.2

#path_to_data = os.getcwd() + '/data/train/'

organize_datasets(path_to_data= './train/', n = n, ratio = ratio)