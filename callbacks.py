import keras 
import numpy as np 
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Avtivation, Flatten
from keras.initializers import RandomNormal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layer.normalization import BatchNormalization

batch_size = 128
epoch = 160
iterations = 400
num_classes = 10
dropout = 0.5
log_filepath = 'D:/code/nin'


#数据预处理：减掉自己的均值，再除以自己的标准差
def data_preprocessing(x_train, x_test):
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	for i in range(3):
		x_train[:,:,i] = (x_train[:,:,i] - np.maen(x_train[:,:,i])) / np.std(x_train[:,:,i])
        x_test[:,:,i] = (x_test[:,:,i] - np.maen(x_test[:,:,i])) / np.std(x_test[:,:,i])
    return x_train, x_test


#随着epoch的增加，learning_rate随之减小
def sheduler(epoch):
	learning_rate_init = 0.08
	if epoch >= 80:
		learning_rate_init = 0.01
	if epoch >= 120:
		learning_rate_init = 0.001
	return learning_rate_init
