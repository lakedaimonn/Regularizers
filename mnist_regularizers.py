# https://snowdeer.github.io/machine-learning/2018/01/09/recognize-mnist-data/

from keras.datasets import mnist # 당연 코드
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Flatten, Activation
from keras import regularizers
from keras.regularizers import l2

from keras import models
from keras import layers

(X_train, Y_train), (X_validation, Y_validation) = mnist.load_data() #당연 코드

for x in X_train[0]:
  for i in x:
    print('{:3} '.format(i), end='')
  print()

X_train = X_train.reshape(X_train.shape[0], 784).astype('float64') / 255 # 28*28 = 784 (데이터를 한 줄로 바꿨다는 뜻), astype: 있어도 그만 없어도 그만. float: 실수. /255: 스케일링
X_validation = X_validation.reshape(X_validation.shape[0], 784).astype( # 검증 데이터
    'float64') / 255

Y_train = np_utils.to_categorical(Y_train, 10) # to_categorical: one-hot-encoding. 0/1으로 딱 떨어지게 만드는 것.
Y_validation = np_utils.to_categorical(Y_validation, 10)
# 이미지를 스케일링하고 one-hot-encoding하는 것

model = Sequential()
model.add(Dense(1400, input_dim=784, activation='relu')) # input이 784, output이 512. output 개수가 무엇이든 상관 X
model.add(layers.Dense(64, kernel_regularizer=l2(0.01), activation='relu'))
# model.add(Dense(512, activation='relu)) # 1400 입력
# model.add(Dense(400, activation='relu)) 3# 512 입력
# model.add(Dense(100, activation='relu))
model.add(Dense(10, activation='softmax')) # 손글씨 숫자는 0-9까지 이므로 10.

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

#model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=30, batch_size=200, verbose=0)

print('\nAccuracy: {:.4f}'.format(model.evaluate(X_validation, Y_validation)[1])) # 한 번만 프린트해라


history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=30, batch_size=500)

import matplotlib.pyplot as plt


loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

epochs = range(1, len(loss) + 1)


plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()
