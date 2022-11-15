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


import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from keras.callbacks import ModelCheckpoint
from scipy import stats


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM


from keras.utils import plot_model

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding

from keras.utils import plot_model


class Actions(Enum):




     kick = 0
     punch = 1
     squat = 2
     stand = 3
     attention = 4
     cancel = 5
     walk = 6
     Sit = 7
     Direction = 8
     PhoneCall=9








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
                          normalize=True,
                          title='Normalized Confusion Matrix',
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
raw_data = pd.read_csv('your csv datasets', header=0)
dataset = raw_data.values
print(dataset)

#X = dataset[:, 0:36].astype(float)
#Y = dataset[:, 36]

X = dataset[0:9869, 0:36].astype(float)
Y = dataset[0:9869, 36]

print(X)
print(Y)


#encoder_Y = [0]*744 + [1]*722 + [2]*668 + [3]*692
#encoder_Y = [0]*1100 + [1]*2176 + [2]*1170 + [3]*2030 + [4]*1200

#encoder_Y = [0]*784 + [1]*583 + [2]*711 + [3]*907 + [4]*1623+ [5]*1994+ [6]*722+ [7]*942
#encoder_Y = [0]*984 + [1]*1163 + [2]*929 + [3]*906 + [4]*1622+ [5]*1993+ [6]*1021+ [7]*941+ [8]*961+ [9]*1041+ [10]*999

#encoder_Y =[0]*784 + [1]*583 + [2]*711 + [3]*907 + [4]*1000 + [5]*1100 + [6]*1000 + [7]*1000+ [8]*1100+ [9]*1500+ [10]*1500

#encoder_Y =[0]*1000 + [1]*1100 + [2]*1000 + [3]*1000 + [4]*1100 + [5]*1100 + [6]*1500 + [7]*1700+ [8]*1500

encoder_Y = [0]*784 + [1]*583 + [2]*711 + [3]*907 + [4]*1623+ [5]*1994+ [6]*722+ [7]*942+ [8]*962+ [9]*641
print(encoder_Y)
# one hot 
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


plot_model(model, to_file='model_test.png', show_shapes=True)

# training
his = LossHistory()

model.compile(optimizer=Adam(0.0001),loss='categorical_crossentropy', metrics=['accuracy'])

his = LossHistory()


history=model.fit(X_train, Y_train, batch_size=32, epochs=50, verbose=1, validation_data=(X_test, Y_test), callbacks=[his])

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


model.summary()
his.loss_plot('epoch')
model.save('model.h5')
print("Saved model to disk")





# # confusion matrix
Y_pred = model.predict(X_train)
cfm = confusion_matrix(np.argmax(Y_train,axis=1), np.argmax(Y_pred, axis=1))
np.set_printoptions(precision=2)
plt.figure()
#class_names = ['stand', 'walk','squat',  'wave']

class_names = ['kick', 'punch','squat','stand','attention','cancel','walk','sit','direction','PhoneCall']


#class_names = ['Attention', 'Direction','PhoneCall','Ache','Cold','Stand','Squat','Kick','Stop']
plot_confusion_matrix(cfm, classes=class_names, title='Normalized Confusion Matrix')
plt.show()




# # evaluate and draw confusion matrix
print('Test:')
score, accuracy = model.evaluate(X_test,Y_test,batch_size=32)
print('Test Score:{:.3}'.format(score))
print('Test accuracy:{:.3}'.format(accuracy))

his.loss_plot('epoch')

# # confusion matrix
Y_pred = model.predict(X_test)
cfm = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(Y_pred, axis=1))
np.set_printoptions(precision=2)
plt.figure()
#class_names = ['stand', 'walk','squat',  'wave']

class_names = ['kick', 'punch','squat','stand','attention','cancel','walk','sit','direction','PhoneCall']


#class_names = ['Attention', 'Direction','PhoneCall','Ache','Cold','Stand','Squat','Kick','Stop']
plot_confusion_matrix(cfm, classes=class_names, title='Normalized Confusion Matrix')
plt.show()

# # test

model = load_model('model.h5')

test_input = [0.43, 0.46, 0.43, 0.52, 0.4, 0.52, 0.39, 0.61, 0.4,
              0.67, 0.46, 0.52, 0.46, 0.61, 0.46, 0.67, 0.42, 0.67,
           0.42, 0.81, 0.43, 0.91, 0.45, 0.67, 0.45, 0.81, 0.45,
            0.91, 0.42, 0.44, 0.43, 0.44, 0.42, 0.46, 0.44, 0.46]
test_np = np.array(test_input)
test_np = test_np.reshape(-1, 36)

test_np = np.array(X[1033]).reshape(-1, 36)
if test_np.size > 0:
  pred = np.argmax(model.predict(test_np))
  init_label = Actions(pred).name

print(init_label)


