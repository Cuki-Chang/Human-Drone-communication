import pandas as pd
from enum import Enum
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model

import matplotlib.pyplot as plt
from keras.callbacks import Callback
import itertools
from sklearn.metrics import confusion_matrix


from keras import optimizers
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


class Actions(Enum):

     Attention=0
     Direction=1
     PhoneCall = 2
     Ache= 3
     Cold = 4
     Stand =5
     Squat=6
     Kick=7
     Stop =8

     Cancel=10

# Callback class to visialize training progress
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# load data
raw_data = pd.read_csv('labeldata10.csv', header=0)
dataset = raw_data.values
print(dataset)

X = dataset[0:12500, 0:36].astype(float)
Y = dataset[0:12500, 36]

print(X)
print(Y)
# 将类别编码为数字


encoder_Y =[0]*1000 + [1]*1100 + [2]*1000 + [3]*1000 + [4]*1100 + [5]*1100 + [6]*1500 + [7]*1700+ [8]*1500+ [9]*1500
print(encoder_Y)
# one hot 编码
dummy_Y = np_utils.to_categorical(encoder_Y)
print(dummy_Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,dummy_Y, test_size=0.1, random_state=9)

# build keras model
model = Sequential()
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=10, activation='softmax'))  # units = nums of classes



# training
his = LossHistory()

rms = optimizers.RMSprop(lr=0.001, epsilon=1e-8, rho=0.9)
model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])

#model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy', metrics=['accuracy'])

#model.fit(X_train,Y_train, batch_size=20, epochs=2, shuffle=True)
model.fit(X_train, Y_train, batch_size=32, epochs=1, verbose=1, validation_data=(X_test, Y_test), callbacks=[his])
model.summary()
# test the model
score = model.evaluate(X_test, Y_test, batch_size=32)

print ('loss:\t', score[0], '\naccuracy:\t', score[1])

